"""Shared raw-mask protocol for MSSL and its newly trained control.

Single-original-box images use the original semantic mask, threshold >127.
Multiple-original-box images retain existing native labels and are ineligible
for the extra loss even when LSJ later removes all but one box.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_util
from detectron2.data.datasets import load_coco_json
from polypbench.adapters.bitmask_mapper import BitmaskLSJMapper


def load_records(annotations, image_root, dataset_name, raw_mask_root):
    records = load_coco_json(annotations, image_root, dataset_name)
    for record in records:
        single = len(record["annotations"]) == 1
        record["mssl_source_single"] = single
        if single:
            path = Path(raw_mask_root) / Path(record["file_name"]).name
            with Image.open(path) as image:
                mask = np.asarray(image.convert("L")) > 127
            if mask.shape != (record["height"], record["width"]):
                raise ValueError(f"Mask/image size mismatch: {path}")
            rle = mask_util.encode(np.asfortranarray(mask, dtype=np.uint8))
            record["annotations"][0]["segmentation"] = rle
    return records


class MSSLMapper(BitmaskLSJMapper):
    def __call__(self, original):
        mapped = super().__call__(original)
        instances = mapped["instances"]
        instances.gt_mssl_eligible = torch.full(
            (len(instances),), bool(original["mssl_source_single"]), dtype=torch.bool)
        return mapped
