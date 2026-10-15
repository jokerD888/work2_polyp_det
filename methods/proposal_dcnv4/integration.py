"""Compose independent head/SCDA switches, keeping every old file untouched."""
import hashlib
from pathlib import Path

from methods.scda_proposal.integration import NATIVE_COMMIT

# Additional insertion-point contract; SCDA checks its own native source files.
HEAD_SHA256 = "248d1b6180766aea2116d484b1756db0e2a1dea76ab292ccd3df5f09bb7a8c2b"


def register(native_repo):
    from methods.scda_proposal.integration import register as register_scda
    register_scda(native_repo)
    path = Path(native_repo).resolve() / "maskdino/modeling/meta_arch/maskdino_head.py"
    actual = hashlib.sha256(path.read_text(encoding="utf-8").replace("\r\n", "\n").encode()).hexdigest()
    if actual != HEAD_SHA256:
        raise RuntimeError("Native MaskDINOHead changed; review DCNv4 insertion point")
    from . import dcnv4_head  # noqa: F401; no DCNv4 CUDA import at registration


def configure_model(cfg):
    from methods.scda_proposal.integration import configure_model as configure_scda
    from .config import DCNv4Options
    options = DCNv4Options.from_cfg(cfg)
    scda = configure_scda(cfg)
    if options.enabled:
        if cfg.MODEL.SEM_SEG_HEAD.NAME != "MaskDINOHead":
            raise ValueError("Adapter expects the unmodified native MaskDINOHead")
        if cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM != options.channels:
            raise ValueError("Adapter channels differ from native CONVS_DIM")
        cfg.MODEL.SEM_SEG_HEAD.NAME = "ProposalDCNv4Head"
    return options, scda
