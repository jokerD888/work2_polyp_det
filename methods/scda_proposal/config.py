"""Non-learned, versioned assignment options; native prediction is unchanged."""
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
    reg_selection: str = "coverage_topk"  # F-v1 compatibility, explicit F-v2 preset
    cls_loss_form: str = "bce"  # "bce" (F-v1/v2) or "focal_positive" (native-aligned)
    focal_alpha: float = 0.25  # mirrors native criterion.py sigmoid_focal_loss constants
    focal_gamma: float = 2.0
    box_gate_rho: float | None = None  # populated ONLY by a verified train calibration

    def validate(self):
        if type(self.enabled) is not bool or type(self.diagnostics) is not bool:
            raise ValueError("ENABLED and DIAGNOSTICS must be booleans")
        if not 0 <= self.tau_reg <= self.tau_cls <= 1:
            raise ValueError("Require 0 <= TAU_REG <= TAU_CLS <= 1")
        if any(type(k) is not int or k <= 0 for k in (self.k_cls, self.k_reg)):
            raise ValueError("K_CLS and K_REG must be positive integers")
        if any(not math.isfinite(w) or w < 0 for w in (self.lambda_cls, self.lambda_box)):
            raise ValueError("Auxiliary loss weights must be finite and nonnegative")
        if self.cls_loss_form not in ("bce", "focal_positive"):
            raise ValueError("Unknown auxiliary classification loss form")
        if any(not math.isfinite(v) or v < 0 for v in (self.focal_alpha, self.focal_gamma)):
            raise ValueError("Focal constants must be finite and nonnegative")
        if self.enabled and self.mask_policy not in ("single_instance", "explicit"):
            raise ValueError("SCDA mask reliability policy needs explicit confirmation")
        if self.enabled and self.geometry != "padded_grid_exact_area_box_distance":
            raise ValueError("SCDA cell geometry and distance convention need confirmation")
        if self.padding_policy != "full_cell_valid":
            raise ValueError("SCDA discards any cell containing padding")
        if self.reg_selection not in ("coverage_topk", "core_plus_transition", "quality_iou", "quality_iou_boxonly", "quality_iou_boxgate"):
            raise ValueError("Unknown regression selection policy")
        if self.reg_selection == "quality_iou_boxgate":
            if (type(self.box_gate_rho) not in (int, float) or
                    not math.isfinite(self.box_gate_rho) or not 0 < self.box_gate_rho <= 1):
                raise ValueError("Box-Gated IoU needs a train-calibrated box_gate_rho in (0,1]")
            if self.mask_policy != "single_instance" or self.lambda_cls != 0 or self.lambda_box != .05:
                raise ValueError("Confirmed Box-Gated is single-instance, lambda_cls=0, lambda_box=.05")
        elif self.box_gate_rho is not None:
            raise ValueError("box_gate_rho is only valid for quality_iou_boxgate")
        if self.reg_selection == "core_plus_transition" and (
            self.k_cls != 2 or self.k_reg != 4 or self.tau_cls != .75 or self.tau_reg != .25
        ):
            raise ValueError("Confirmed F-v2 requires cls2 + reg(core2/transition2), thresholds .75/.25")
        if self.reg_selection in ("quality_iou", "quality_iou_boxonly", "quality_iou_boxgate") and (
            self.k_cls != 2 or self.k_reg != 4 or self.tau_cls != .75 or self.tau_reg != .25
        ):
            raise ValueError("quality_iou presets keep the confirmed pool and quotas: cls2 + reg4, thresholds .75/.25")
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
