"""Explicit V1 choices; these are research defaults, not official MaskDINO defaults."""
from dataclasses import asdict, dataclass
import math
from pathlib import Path

from polypbench.config import load_config


@dataclass(frozen=True)
class MSSLOptions:
    enabled: bool = True
    support: str = "mask"
    weight: float = 1.0
    temperature: float = 0.05
    direction_bandwidth: float = 0.15
    min_scale_pixels: float = 4.0
    loss_form: str = "cdf"

    def validate(self):
        if type(self.enabled) is not bool or self.support not in ("mask", "box"):
            raise ValueError("MSSL requires a bool enabled and mask/box support")
        if self.loss_form not in ("cdf", "static"):
            raise ValueError("loss_form must be cdf or static")
        for key in ("weight", "temperature", "direction_bandwidth", "min_scale_pixels"):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0 or (key != "weight" and value == 0):
                raise ValueError(f"Invalid MSSL {key}")
        return self


def load_experiment(path):
    cfg = load_config(path)
    cfg["mssl"] = asdict(MSSLOptions(**cfg.get("mssl", {})).validate())
    cfg["dataset"]["raw_mask_root"] = str((Path(cfg["project_root"]) /
                                              cfg["dataset"]["raw_mask_root"]).resolve())
    if not Path(cfg["dataset"]["raw_mask_root"]).is_dir():
        raise ValueError("Raw Kvasir mask directory missing")
    micro = cfg["runtime"].get("micro_batch", 1)
    batch = cfg["overrides"]["SOLVER.IMS_PER_BATCH"]
    if type(micro) is not int or micro < 1 or batch % micro:
        raise ValueError("micro_batch must divide the effective total batch")
    if cfg["runtime"]["num_gpus"] != 1:
        raise ValueError("V1 accumulation is validated for one GPU only")
    cfg["runtime"]["micro_batch"] = micro
    return cfg


def add_mssl_config(cfg, options):
    from detectron2.config import CfgNode
    cfg.MODEL.MSSL = CfgNode()
    for key, value in options.items():
        setattr(cfg.MODEL.MSSL, key.upper(), value)


def build_cfg(experiment, output, device=None):
    import train_net
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from .model import register
    register()
    cfg = get_cfg()
    add_deeplab_config(cfg)
    train_net.add_maskdino_config(cfg)
    add_mssl_config(cfg, experiment["mssl"])
    cfg.merge_from_file(experiment["model"]["config"])
    for key, value in experiment["overrides"].items():
        cfg.merge_from_list([key, repr(value)])
    cfg.MODEL.META_ARCHITECTURE = "MSSLMaskDINO"
    if "box_regression" in experiment:
        from methods.ciou_replace.config import validate
        from methods.ciou_replace.model import CIoUReplaceMaskDINO
        validate(experiment)
        cfg.MODEL.META_ARCHITECTURE = "CIoUReplaceMaskDINO"
    if "quality_alignment" in experiment:
        from methods.quality_alignment.config import validate
        from methods.quality_alignment.model import QualityAlignMaskDINO
        validate(experiment)
        cfg.MODEL.META_ARCHITECTURE = "QualityAlignMaskDINO"
        cfg.MODEL.QUALITY_ALIGNMENT_ENABLED = experiment["quality_alignment"]["enabled"]
    cfg.DATASETS.TRAIN, cfg.DATASETS.TEST = ("bench_train",), ("bench_val",)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = experiment["model"]["num_classes"]
    cfg.MODEL.WEIGHTS = experiment["model"]["weights"]
    cfg.OUTPUT_DIR, cfg.SEED = str(output), experiment["seed"]
    cfg.TEST.AUG.ENABLED = False
    cfg.TEST.DETECTIONS_PER_IMAGE = experiment["model"].get("detections_per_image", 100)
    if device:
        cfg.MODEL.DEVICE = device
    cfg.freeze()
    return cfg
