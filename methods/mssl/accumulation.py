"""Single-GPU AMP accumulation; microbatch protocol is explicitly recorded.

Each microbatch keeps native criterion normalization and DN construction.
Consequently this preserves effective image batch and optimizer step budget,
but is not mathematically identical to a physical batch of 16 mixed instances.
Both the method and control must use the same microbatch configuration.
"""
import time
import torch
from detectron2.engine import AMPTrainer
from detectron2.utils.events import get_event_storage


class AccumulatingAMPTrainer(AMPTrainer):
    def __init__(self, model, data_loader, optimizer, accumulation_steps, probe=None):
        super().__init__(model, data_loader, optimizer)
        self.accumulation_steps = accumulation_steps
        self.successful_updates = 0
        self.probe = probe

    def run_step(self):
        assert self.model.training
        self.optimizer.zero_grad(set_to_none=True)
        means = {}
        data_time = 0.
        eligible = 0
        side_sums = [0.] * 4
        quality_means = {}
        scope_means = {}
        for micro_index in range(self.accumulation_steps):
            start = time.perf_counter()
            data = next(self._data_loader_iter)
            data_time += time.perf_counter() - start
            if self.probe is not None:
                self.model.criterion.probe_callback = self.probe.callback(self.iter, micro_index)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss_dict = self.model(data)
                loss = sum(loss_dict.values()) / self.accumulation_steps
            if self.probe is not None:
                self.model.criterion.probe_callback = None
            if not torch.isfinite(loss).all():
                raise FloatingPointError(f"Nonfinite training loss at optimizer iteration {self.iter}")
            self.grad_scaler.scale(loss).backward()
            diagnostics = getattr(self.model.criterion, "last_diagnostics", None)
            if diagnostics:
                eligible += diagnostics["eligible"]
                side_sums = [a+b for a,b in zip(side_sums,diagnostics["side_loss"])]
            for key, value in getattr(self.model.criterion, "last_q_stats", {}).items():
                quality_means[key] = quality_means.get(key, 0.) + value / self.accumulation_steps
            for key, value in getattr(self.model.criterion, "last_scope_stats", {}).items():
                scope_means[key] = scope_means.get(key, 0.) + value / self.accumulation_steps
            for key, value in loss_dict.items():
                means[key] = means.get(key, 0.) + value.detach() / self.accumulation_steps
        self.after_backward()
        self._write_metrics(means, data_time)
        previous_scale = self.grad_scaler.get_scale()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        skipped = self.grad_scaler.get_scale() < previous_scale
        self.successful_updates += int(not skipped)
        storage = get_event_storage()
        storage.put_scalar("amp/loss_scale", self.grad_scaler.get_scale(), smoothing_hint=False)
        storage.put_scalar("amp/skipped_step", int(skipped), smoothing_hint=False)
        storage.put_scalar("optimizer/successful_updates", self.successful_updates, smoothing_hint=False)
        for key, value in quality_means.items():
            storage.put_scalar("quality/" + key, value)
        for key, value in scope_means.items():
            storage.put_scalar("mssl_scope/" + key, value)
        if hasattr(self.model.criterion,"last_diagnostics"):
            storage.put_scalar("mssl/eligible_per_microbatch",eligible/self.accumulation_steps)
            for name,value in zip(("left","top","right","bottom"),side_sums):
                storage.put_scalar(f"mssl/side_loss_{name}",value/self.accumulation_steps)
