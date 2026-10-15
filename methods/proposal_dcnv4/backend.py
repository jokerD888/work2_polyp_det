"""Load only the pinned official implementation; never substitute another op."""
import hashlib
import importlib
from pathlib import Path

DCNV4_COMMIT = "4b848f7dd7da74ff03f7d278f902c6fd05b391b5"
DCNV4_URL = "https://github.com/OpenGVLab/DCNv4"
SOURCE_HASHES = {
    "modules/dcnv4.py": "01035cc9429b65cb37c2556f1a4dfa9bfe4cb20574be305be2de848e3ba011f1",
    "functions/dcnv4_func.py": "ed3dc365bb5e031e1a9508a9e8abd0277d9065b0cfdc6b4559dca8e8418143cc",
}


def source_identity(root):
    root = Path(root).resolve()
    hashes = {}
    for relative, expected in SOURCE_HASHES.items():
        path = root / relative
        actual = hashlib.sha256(path.read_text(encoding="utf-8").replace("\r\n", "\n").encode()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Unreviewed DCNv4 source: {path}; expected commit {DCNV4_COMMIT}")
        hashes[relative] = actual
    return {"url": DCNV4_URL, "commit": DCNV4_COMMIT, "package": str(root), "sources": hashes}


def official_class():
    try:
        module = importlib.import_module("DCNv4.modules.dcnv4")
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "Official DCNv4 extension unavailable. Build the pinned source in an isolated "
            "model environment using README.md. No CPU/Conv/DCNv2 fallback is implemented."
        ) from error
    root = Path(module.__file__).resolve().parents[1]
    identity = source_identity(root)
    extension = importlib.import_module("DCNv4.ext")
    extension_path = Path(extension.__file__).resolve()
    identity["extension"] = str(extension_path)
    identity["extension_sha256"] = hashlib.sha256(extension_path.read_bytes()).hexdigest()
    return module.DCNv4, identity
