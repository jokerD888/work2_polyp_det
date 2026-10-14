import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_api


@unittest.skipUnless(importlib.util.find_spec("detectron2") and importlib.util.find_spec("torch"), "Requires the Linux detector environment")
class MapperTests(unittest.TestCase):
    def test_annotation_boxes_survive_loading_when_masks_have_different_extents(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "MaskDINO"))
        from detectron2.config import get_cfg
        from detectron2.structures import BoxMode
        from polypbench.adapters.bitmask_mapper import BitmaskLSJMapper
        cfg = get_cfg()
        cfg.INPUT.IMAGE_SIZE = 16
        cfg.INPUT.MIN_SCALE = 1.0
        cfg.INPUT.MAX_SCALE = 1.0
        cfg.INPUT.RANDOM_FLIP = "none"
        binary = np.zeros((16, 16), dtype=np.uint8)
        binary[3:10, 3:10] = 1
        rle = mask_api.encode(np.asfortranarray(binary))
        rle["counts"] = rle["counts"].decode("ascii")
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "image.png"
            Image.new("RGB", (16, 16), "red").save(file)
            record = {"file_name": str(file), "height": 16, "width": 16, "annotations": [{"category_id": 0, "bbox": [1, 1, 14, 14], "bbox_mode": BoxMode.XYXY_ABS, "segmentation": rle, "iscrowd": 0}]}
            preserved = BitmaskLSJMapper(cfg, training_boxes="annotations")(record)
            np.testing.assert_array_equal(preserved["instances"].gt_boxes.tensor.numpy(), [[1, 1, 14, 14]])
            np.testing.assert_array_equal(preserved["instances"].gt_masks.numpy(), binary[None].astype(bool))
            legacy = BitmaskLSJMapper(cfg, training_boxes="mask")(record)
            np.testing.assert_array_equal(legacy["instances"].gt_boxes.tensor.numpy(), [[3, 3, 10, 10]])


if __name__ == "__main__":
    unittest.main()
