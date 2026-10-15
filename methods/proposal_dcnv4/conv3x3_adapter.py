"""Zero-initialized ordinary Conv control; no DCNv4 backend imports."""
import torch
from torch import nn

from .config import DCNv4Options


class Conv3x3Residual(nn.Module):
    def __init__(self, options: DCNv4Options):
        super().__init__()
        self.options = options.validate()
        if options.operator != "conv3x3":
            raise ValueError("Conv3x3Residual requires operator=conv3x3")
        with torch.random.fork_rng(devices=[]):
            self.conv = nn.Conv2d(options.channels, options.channels, kernel_size=3,
                                  stride=1, padding=1, dilation=1, groups=1, bias=True).float()
            nn.init.zeros_(self.conv.weight)
            nn.init.zeros_(self.conv.bias)
        self.alpha = float(options.alpha)
        self.backend_identity = {
            "kind": "torch.nn.Conv2d", "operator": "conv3x3", "groups": 1,
            "kernel_size": 3, "stride": 1, "padding": 1,
            "initialization": "weight=0,bias=0", "depends_on_dcnv4": False,
        }

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.options.channels:
            raise ValueError(f"Expected BCHW with {self.options.channels} channels; got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise TypeError("Conv control features must be floating point")
        if self.alpha == 0:
            return x
        if any(parameter.dtype != torch.float32 for parameter in self.conv.parameters()):
            raise TypeError("Keep Conv control parameters FP32; use autocast instead of model.half()")
        with torch.autocast(device_type=x.device.type, enabled=False):
            delta = self.conv(x.float())
            return x + self.alpha * delta.to(dtype=x.dtype)
