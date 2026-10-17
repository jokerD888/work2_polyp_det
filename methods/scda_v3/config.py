"""Explicit V3 policies; the legacy SCDA configuration is never rewritten."""
from dataclasses import asdict, dataclass
import math

from methods.scda_proposal.config import SCDAOptions


@dataclass(frozen=True)
class V3Options:
    enabled: bool = True
    assignment_policy: str = "legacy_quality_iou"
    lambda_aux: float = .05
    tau_reg: float = .25
    k_reg: int = 4
    mask_policy: str = "single_instance"
    diagnostics: bool = True
    target_cells: float = 8.0
    preferred_levels: int = 2

    def validate(self):
        if type(self.enabled) is not bool or type(self.diagnostics) is not bool:
            raise ValueError("enabled/diagnostics must be bool")
        if self.assignment_policy not in ("legacy_quality_iou", "mask_multiscale"):
            raise ValueError("Unknown V3 assignment policy")
        if not math.isfinite(self.lambda_aux) or self.lambda_aux < 0:
            raise ValueError("lambda_aux must be finite and nonnegative")
        if (self.tau_reg != .25 or type(self.k_reg) is not int or self.k_reg != 4 or
                self.mask_policy != "single_instance" or self.target_cells != 8 or
                type(self.preferred_levels) is not int or self.preferred_levels != 2):
            raise ValueError("Confirmed V3 keeps coverage .25, K4, single-instance, 8 cells and 2 levels")
        return self

    def legacy(self, diagnostics=None):
        return SCDAOptions(enabled=True, reg_selection="quality_iou", lambda_cls=0.,
                           lambda_box=self.lambda_aux, diagnostics=(self.diagnostics if diagnostics is None
                                                                  else diagnostics)).validate()

    @classmethod
    def from_cfg(cls, cfg):
        return cls(**{key: getattr(cfg.MODEL.SCDA_V3, key.upper()) for key in asdict(cls())}).validate()


def add_v3_config(cfg):
    from detectron2.config import CfgNode
    cfg.MODEL.SCDA_V3 = CfgNode()
    for key, value in asdict(V3Options()).items():
        setattr(cfg.MODEL.SCDA_V3, key.upper(), value)


def build_cfg(experiment, output, device=None):
    """One resolved configuration path for the worker and acceptance tools."""
    import train_net
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from .integration import configure_model
    cfg = get_cfg()
    add_deeplab_config(cfg)
    train_net.add_maskdino_config(cfg)
    add_v3_config(cfg)
    cfg.merge_from_file(experiment["model"]["config"])
    for key, value in experiment["overrides"].items():
        cfg.merge_from_list([key, repr(value)])
    for key, value in experiment["v3"].items():
        setattr(cfg.MODEL.SCDA_V3, key.upper(), value)
    configure_model(cfg)
    cfg.DATASETS.TRAIN, cfg.DATASETS.TEST = ("bench_train",), ("bench_val",)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = experiment["model"]["num_classes"]
    cfg.MODEL.WEIGHTS = experiment["model"]["weights"]
    cfg.OUTPUT_DIR, cfg.SEED = str(output), experiment["seed"]
    cfg.TEST.AUG.ENABLED = False
    cfg.TEST.DETECTIONS_PER_IMAGE = experiment["model"].get("detections_per_image", 100)
    if device is not None:
        cfg.MODEL.DEVICE = device
    cfg.freeze()
    return cfg
