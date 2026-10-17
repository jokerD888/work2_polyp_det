"""Pinned native contract and separate V3 class names, with no monkeypatching."""
from methods.scda_proposal.integration import NATIVE_COMMIT, register as register_legacy
from .config import V3Options


def register(native_repo):
    register_legacy(native_repo)
    from . import decoder, model  # noqa: F401


def configure_model(cfg):
    options = V3Options.from_cfg(cfg)
    if options.enabled:
        if (cfg.MODEL.META_ARCHITECTURE != "MaskDINO" or
                cfg.MODEL.MaskDINO.TRANSFORMER_DECODER_NAME != "MaskDINODecoder" or
                not cfg.MODEL.MaskDINO.TWO_STAGE or cfg.MODEL.MaskDINO.SEMANTIC_CE_LOSS or
                cfg.INPUT.DATASET_MAPPER_NAME != "coco_instance_lsj"):
            raise ValueError("V3 requires pinned sigmoid TWO_STAGE MaskDINO + LSJ, without other adapters")
        cfg.MODEL.META_ARCHITECTURE = "SCDAv3MaskDINO"
        cfg.MODEL.MaskDINO.TRANSFORMER_DECODER_NAME = "SCDAv3Decoder"
    return options
