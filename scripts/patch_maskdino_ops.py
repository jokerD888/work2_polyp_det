"""Explicit compatibility patch for old ATen dispatch under PyTorch 2.7.

Only replaces removed API spellings; equations and CUDA kernels remain native.
The checkout diff is recorded with each experiment.
"""
import argparse
from pathlib import Path


def patch(repo):
    file = Path(repo) / "maskdino/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu"
    source = file.read_text()
    old = 'AT_DISPATCH_FLOATING_TYPES(value.type(),'
    new = 'AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),'
    if old not in source and source.count(new) != 2:
        raise RuntimeError("Unexpected upstream source; review compatibility patch manually")
    file.write_text(source.replace(old, new))
    print(f"Patched ATen dispatch API: {file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="third_party/MaskDINO")
    patch(parser.parse_args().repo)
