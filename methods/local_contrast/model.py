import json
from pathlib import Path
import torch
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from methods.mssl.model import MSSLMaskDINO
from .config import Options
from .regions import plan
from .loss import region_loss


@META_ARCH_REGISTRY.register()
class LocalContrastMaskDINO(MSSLMaskDINO):
    """Direct contrast on shared Encoder tokens, with no additional parameters.

    A pre-hook reads the exact tensors passed into the native detection
    predictor. It does not replace or detach any detection-path tensor.
    """
    @configurable
    def __init__(self, *, local_contrast_options, **kwargs):
        super().__init__(**kwargs)
        self.local_options = Options(**local_contrast_options).validate()
        self.local_sequence = 0
        self.local_feature = None
        self.last_local_stats = {}
        self.local_probe_path = None
        self.local_probe_active = False
        self.sem_seg_head.predictor.register_forward_pre_hook(self._capture)

    @classmethod
    def from_config(cls, cfg):
        result = super().from_config(cfg)
        result['local_contrast_options'] = dict(cfg.MODEL.LOCAL_CONTRAST)
        return result

    def _capture(self, module, args):
        if self.training and self.local_options.variant != 'off':
            candidates = [x for x in args[0] if tuple(x.shape[-2:]) == self.local_expected_hw]
            if len(candidates) != 1:
                raise RuntimeError('Expected one actual stride8 Encoder feature in detection input')
            self.local_feature = candidates[0]

    def forward(self, batched_inputs):
        if not self.training or self.local_options.variant == 'off':
            return super().forward(batched_inputs)
        if len(batched_inputs) != 1:
            raise ValueError('Local contrast protocol requires microbatch1')
        image = batched_inputs[0]
        self.local_expected_hw = tuple(s//8 for s in image['image'].shape[-2:])
        rows, stats = plan(image, self.local_options, self.local_sequence)
        self.local_sequence += 1
        self.local_feature = None
        try:
            result = super().forward(batched_inputs)
            feature = self.local_feature
            if feature is None or not feature.requires_grad:
                raise RuntimeError('Detection Encoder feature was not captured with autograd')
            extra = region_loss(feature[0], rows, self.local_options.temperature)
            weighted = extra*self.local_options.weight
            self.last_local_stats = dict(stats, unweighted_loss=float(extra.detach()),
                                         weighted_loss=float(weighted.detach()))
            if self.local_probe_active:
                self._probe(weighted, feature, result)
            result['loss_local_contrast'] = weighted
            return result
        finally:
            self.local_feature = None

    def _probe(self, weighted, feature, native_losses):
        # Read-only unscaled diagnostic before global gradient clipping.
        probes = dict(local=weighted, native_total=sum(native_losses.values()))
        record = dict(sequence=self.local_sequence-1, stats=self.last_local_stats, gradients={})
        for name, loss in probes.items():
            gradient = torch.autograd.grad(loss*64, feature, retain_graph=True, allow_unused=True)[0]
            finite = gradient is not None and bool(torch.isfinite(gradient).all())
            record['gradients'][name] = dict(finite=finite,
                feature_l2=float((gradient.detach().float()/64).norm()) if finite else None)
        if self.local_probe_path:
            with Path(self.local_probe_path).open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False)+'\n')
        self.last_local_probe = record
