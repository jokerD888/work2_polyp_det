"""BCHW -> official B(HW)C operator -> fixed-alpha residual, in FP32."""
import torch
from torch import nn

from .config import DCNv4Options


class DCNv4Residual(nn.Module):
    def __init__(self, options: DCNv4Options, *, operator_factory=None):
        super().__init__()
        self.options = options.validate()
        self.backend_identity = {"kind": "injected_test_operator", "cuda_validated": False}
        self._official = operator_factory is None
        if self._official:
            from .backend import official_class
            operator_factory, self.backend_identity = official_class()
        # The official constructor initializes parameters on CPU. Preserve the
        # caller's RNG stream so new parameters do not shift later native
        # initialization or data/augmentation randomness in paired experiments.
        with torch.random.fork_rng(devices=[]):
            self.dcn = operator_factory(
                channels=options.channels, kernel_size=options.kernel_size,
                stride=1, pad=1, dilation=options.dilation, group=options.group,
                offset_scale=options.offset_scale, dw_kernel_size=None,
                center_feature_scale=False, remove_center=False,
                output_bias=True, without_pointwise=False,
            ).float()
        # Plain scalar, deliberately not a learnable gate or state_dict buffer.
        self.alpha = float(options.alpha)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.options.channels:
            raise ValueError(f"Expected BCHW with {self.options.channels} channels; got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise TypeError("DCNv4 features must be floating point")
        if self.alpha == 0:
            return x  # exact bypass, including the operator itself
        if self._official and x.device.type != "cuda":
            raise RuntimeError("Official DCNv4 requires CUDA; CPU tests use an explicit test double only")
        if any(p.dtype != torch.float32 for p in self.dcn.parameters()):
            raise TypeError("Keep DCNv4 parameters FP32; use autocast instead of model.half()")
        b, c, h, w = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            tokens = x.float().permute(0, 2, 3, 1).reshape(b, h * w, c).contiguous()
            delta = self.dcn(tokens, shape=(h, w))
            if delta.shape != tokens.shape:
                raise RuntimeError("Official DCNv4 unexpectedly changed the feature shape")
            delta = delta.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
            return x + self.alpha * delta.to(dtype=x.dtype)
