"""Non-learned F-v1 options, separate from native prediction configuration."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class SCDAOptions:
    enabled: bool = False
    tau_cls: float = 0.75
    tau_reg: float = 0.25
    k_cls: int = 2
    k_reg: int = 4
    lambda_cls: float = 0.05
    lambda_box: float = 0.05
    mask_policy: str = "single_instance"
    geometry: str = "padded_grid_exact_area_box_distance"
    diagnostics: bool = False
    padding_policy: str = "full_cell_valid"

    def validate(self):
        if type(self.enabled) is not bool or type(self.diagnostics) is not bool:
            raise ValueError("ENABLED and DIAGNOSTICS must be booleans")
        if not 0 <= self.tau_reg <= self.tau_cls <= 1:
            raise ValueError("Require 0 <= TAU_REG <= TAU_CLS <= 1")
        if any(type(k) is not int or k <= 0 for k in (self.k_cls, self.k_reg)):
            raise ValueError("K_CLS and K_REG must be positive integers")
        if any(not math.isfinite(w) or w < 0 for w in (self.lambda_cls, self.lambda_box)):
            raise ValueError("Auxiliary loss weights must be finite and nonnegative")
        if self.enabled and self.mask_policy not in ("single_instance", "explicit"):
            raise ValueError("SCDA mask reliability policy needs explicit confirmation")
        if self.enabled and self.geometry != "padded_grid_exact_area_box_distance":
            raise ValueError("SCDA cell geometry and distance convention need confirmation")
        if self.padding_policy != "full_cell_valid":
            raise ValueError("F-v1 discards any cell containing padding")
        return self

    @classmethod
    def from_cfg(cls, cfg):
        node = cfg.MODEL.SCDA
        return cls(**{key: getattr(node, key.upper()) for key in asdict(cls())}).validate()


def add_scda_config(cfg):
    from detectron2.config import CfgNode
    cfg.MODEL.SCDA = CfgNode()
    for name, value in asdict(SCDAOptions()).items():
        setattr(cfg.MODEL.SCDA, name.upper(), value)
