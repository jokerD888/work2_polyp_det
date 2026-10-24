from pathlib import Path
import json
from polypbench.io import write_json


def native_digest(model):
    from methods.reference_features.monitor import native_digest as digest
    return digest(model)


def optimizer_contract(model, optimizer):
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    counts = {}
    for group in optimizer.param_groups:
        for p in group['params']:
            counts[id(p)] = counts.get(id(p), 0)+1
    if set(counts) != expected or any(n != 1 for n in counts.values()):
        raise RuntimeError('Optimizer must discover every trainable parameter exactly once')
    return dict(status='passed', new_parameter_count=0, all_parameters_discovered_once=True)


def make_hook(output):
    from detectron2.engine import HookBase
    class Monitor(HookBase):
        def before_train(self):
            model = self.trainer.model
            write_json(Path(output)/'local_initialization.json', dict(native_sha256=native_digest(model),
                optimizer=optimizer_contract(model, self.trainer.optimizer)))
            model.local_probe_path = str(Path(output)/'local_gradient_probes.jsonl')

        def before_step(self):
            self.trainer.model.local_probe_active = self.trainer.iter in (0, 10) or self.trainer.iter % 275 == 0

        def after_step(self):
            self.trainer.model.local_probe_active = False

    return Monitor()
