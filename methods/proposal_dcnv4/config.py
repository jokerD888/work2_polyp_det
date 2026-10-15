"""Independent adapter options. Disabled means the original head is retained."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class DCNv4Options:
    enabled: bool = False
    channels: int = 256
    feature_stride: int = 8
    kernel_size: int = 3
    group: int = 4
    dilation: int = 1
    offset_scale: float = 1.0
    alpha: float = 0.1
    force_fp32: bool = True

    def validate(self):
        if type(self.enabled) is not bool or type(self.force_fp32) is not bool:
            raise ValueError("enabled/force_fp32 must be booleans")
        for name in ("channels", "feature_stride", "kernel_size", "group", "dilation"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.channels % self.group or (self.channels // self.group) % 16:
            raise ValueError("Official DCNv4 requires channels/group divisible by 16")
        if self.kernel_size != 3 or self.dilation != 1:
            raise ValueError("Initial protocol fixes kernel=3, stride=1, pad=1, dilation=1")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in (self.alpha, self.offset_scale)):
            raise ValueError("alpha and offset_scale must be finite numbers")
        if self.alpha < 0 or self.offset_scale <= 0:
            raise ValueError("Require alpha >= 0 and offset_scale > 0")
        if not self.force_fp32:
            raise ValueError("Initial protocol requires FP32 DCNv4 execution")
        return self

    @classmethod
    def from_cfg(cls, cfg):
        return cls(**{key: getattr(cfg.MODEL.PROPOSAL_DCNV4, key.upper())
                      for key in asdict(cls())}).validate()


def add_dcnv4_config(cfg):
    from detectron2.config import CfgNode
    cfg.MODEL.PROPOSAL_DCNV4 = CfgNode()
    for key, value in asdict(DCNv4Options()).items():
        setattr(cfg.MODEL.PROPOSAL_DCNV4, key.upper(), value)
