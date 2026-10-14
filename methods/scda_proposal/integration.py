"""Explicit registration/configuration, isolated from the active baseline."""
import hashlib
from pathlib import Path

from .config import SCDAOptions


NATIVE_COMMIT = "3831d8514a3728535ace8d4ecc7d28044c42dd14"
# Normalize only newline encoding when guarding the source contract, so the
# same clone works in Windows/WSL and Linux. Native files are NEVER rewritten.
NATIVE_SOURCE_HASHES = {
    "maskdino/modeling/transformer_decoder/maskdino_decoder.py": "b6a15b63a57ed4613a4b49660b51d6de8414debd5a3609ec03024397405bab77",
    "maskdino/maskdino.py": "ce302f7c43e8c26a339c3eefe664b17321f2a569c9471dfdf5022793ed10fac8",
    "maskdino/utils/utils.py": "3865b7f514cd8590e7d7d9b2b286c0e34f2899f69315f6b2bb230a795c340352",
}


def register(native_repo):
    root = Path(native_repo).resolve()
    for relative, expected in NATIVE_SOURCE_HASHES.items():
        text = (root / relative).read_text(encoding="utf-8").replace("\r\n", "\n")
        actual = hashlib.sha256(text.encode()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Native source contract changed: {relative}; re-review integration before using F")
    from . import decoder, model  # noqa: F401: registers only our new class names
    # Also reject an unrelated maskdino package shadowing the configured clone.
    from maskdino import maskdino as native_model
    if Path(native_model.__file__).resolve() != root / "maskdino" / "maskdino.py":
        raise RuntimeError("Imported maskdino does not come from the configured native repository")


def configure_model(cfg):
    options = SCDAOptions.from_cfg(cfg)
    if options.enabled:
        if cfg.MODEL.META_ARCHITECTURE != "MaskDINO":
            raise ValueError("F-v1 supports the pinned native MaskDINO meta-architecture only")
        if cfg.MODEL.MaskDINO.TRANSFORMER_DECODER_NAME != "MaskDINODecoder":
            raise ValueError("F-v1 expects the pinned native MaskDINODecoder")
        cfg.MODEL.META_ARCHITECTURE = "SCDAProposalMaskDINO"
        cfg.MODEL.MaskDINO.TRANSFORMER_DECODER_NAME = "SCDAProposalDecoder"
    # Disabled mode leaves the original class names/configuration intact.
    return options
