import importlib.util
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_api


@unittest.skipUnless(importlib.util.find_spec("detectron2") and importlib.util.find_spec("torch"), "Requires the Linux detector environment")
class MapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "MaskDINO"))

    def padding_case(self, height, width, scale, count=1):
        from detectron2.config import get_cfg
        from detectron2.structures import BoxMode
        cfg = get_cfg()
        cfg.INPUT.IMAGE_SIZE = 16
        cfg.INPUT.MIN_SCALE = scale
        cfg.INPUT.MAX_SCALE = scale
        cfg.INPUT.RANDOM_FLIP = "none"
        masks, annotations = [], []
        for index in range(count):
            binary = np.zeros((height, width), dtype=np.uint8)
            # Include the original right edge to exercise a foreground/pad edge.
            binary[height // 4:3 * height // 4, width // 4:width] = 1
            rle = mask_api.encode(np.asfortranarray(binary))
            rle["counts"] = rle["counts"].decode("ascii")
            masks.append(binary)
            annotations.append({"id": index + 1, "category_id": 0,
                                "bbox": [width // 4, height // 4, width, 3 * height // 4],
                                "bbox_mode": BoxMode.XYXY_ABS,
                                "segmentation": rle, "iscrowd": 0})
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        file = Path(temp.name) / "image.png"
        pixels = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
        Image.fromarray(pixels).save(file)
        record = {"image_id": 1, "file_name": str(file), "height": height,
                  "width": width, "annotations": annotations}
        return cfg, record, masks

    def check_padding_case(self, height, width, scale, resized_height, resized_width, count=1):
        import torch
        from polypbench.adapters.bitmask_mapper import BitmaskLSJMapper
        from methods.scda_proposal.mapper import SCDALSJMapper
        cfg, record, masks = self.padding_case(height, width, scale, count)
        untouched = copy.deepcopy(record)
        native = BitmaskLSJMapper(cfg, "annotations")(record)
        scda = SCDALSJMapper(cfg, "single_instance")(record)
        expected_padding = np.ones((16, 16), dtype=bool)
        expected_padding[:resized_height, :resized_width] = False
        for mapped in (native, scda):
            np.testing.assert_array_equal(mapped["padding_mask"].numpy(), expected_padding)
            actual_masks = mapped["instances"].gt_masks
            self.assertEqual(tuple(actual_masks.shape), (count, 16, 16))
            self.assertFalse(bool(actual_masks[:, expected_padding].any()))
            self.assertTrue(bool((mapped["image"][:, expected_padding] == 128).all()))
            for index, mask in enumerate(masks):
                expected = np.zeros((16, 16), dtype=bool)
                # ResizeTransform uses nearest-neighbour sampling for masks.
                from detectron2.data.transforms import ResizeTransform
                expected[:resized_height, :resized_width] = ResizeTransform(
                    height, width, resized_height, resized_width).apply_segmentation(mask).astype(bool)
                np.testing.assert_array_equal(actual_masks[index].numpy(), expected)
        for key in ("image", "padding_mask"):
            torch.testing.assert_close(native[key], scda[key], rtol=0, atol=0)
        torch.testing.assert_close(native["instances"].gt_boxes.tensor,
                                   scda["instances"].gt_boxes.tensor, rtol=0, atol=0)
        torch.testing.assert_close(native["instances"].gt_masks, scda["instances"].gt_masks, rtol=0, atol=0)
        self.assertEqual(scda["scda_original_gt_count"], count)
        self.assertEqual(scda["instances"].gt_scda_reliable.tolist(), [count == 1] * count)
        self.assertEqual(record, untouched)
        return native

    def test_shrink_padding_is_background_and_invalid(self):
        self.check_padding_case(16, 16, .5, 8, 8)

    def test_non_square_bottom_and_right_padding(self):
        for args in ((10, 20, 1., 8, 16), (20, 10, 1., 16, 8)):
            with self.subTest(shape=args[:2]):
                self.check_padding_case(*args)

    def test_multiple_instances_do_not_acquire_shared_pad_foreground(self):
        self.check_padding_case(16, 16, .5, 8, 8, count=2)

    def test_empty_instances_still_track_padding(self):
        self.check_padding_case(16, 16, .5, 8, 8, count=0)

    def test_scda_rejects_cells_crossing_real_mapper_padding(self):
        import torch
        from methods.scda_proposal.scda_assigner import fully_valid_cells, rectangle_coverage
        mapped = self.check_padding_case(16, 16, .5, 8, 8)
        cells = torch.tensor([[0., 0., 8., 8.], [4., 4., 12., 12.], [8., 8., 16., 16.]])
        valid = ~mapped["padding_mask"]
        self.assertEqual(fully_valid_cells(valid, cells).tolist(), [True, False, False])
        coverage = rectangle_coverage(mapped["instances"].gt_masks, cells)
        self.assertEqual(float(coverage[0, 2]), 0.)

    def test_correction_preserves_legacy_image_boxes_and_nonpad_masks(self):
        import torch
        from polypbench.adapters.bitmask_mapper import BitmaskLSJMapper
        from maskdino.data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import build_transform_gen
        for scale in (.5, 2.):
            with self.subTest(scale=scale):
                cfg, record, _ = self.padding_case(16, 16, scale)
                fixed = BitmaskLSJMapper(cfg, "annotations")
                legacy = BitmaskLSJMapper(cfg, "annotations")
                legacy.augmentations = build_transform_gen(cfg, True)
                np.random.seed(42)
                actual = fixed(record)
                np.random.seed(42)
                previous = legacy(record)
                torch.testing.assert_close(actual["image"], previous["image"], rtol=0, atol=0)
                torch.testing.assert_close(actual["instances"].gt_boxes.tensor,
                                           previous["instances"].gt_boxes.tensor, rtol=0, atol=0)
                valid = ~actual["padding_mask"]
                torch.testing.assert_close(actual["instances"].gt_masks[:, valid],
                                           previous["instances"].gt_masks[:, valid], rtol=0, atol=0)
                if scale < 1:
                    self.assertTrue(bool(previous["instances"].gt_masks[:, ~valid].all()))
                    self.assertFalse(bool(previous["padding_mask"].any()))
                else:
                    self.assertFalse(bool(actual["padding_mask"].any()))
                    torch.testing.assert_close(actual["instances"].gt_masks,
                                               previous["instances"].gt_masks, rtol=0, atol=0)

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


class MapperSnapshotTests(unittest.TestCase):
    def test_all_runners_track_shared_padding_implementation(self):
        from polypbench.runner import adapter_sources as baseline_sources
        from methods.scda_proposal.run import adapter_sources as scda_sources
        from methods.proposal_dcnv4.run import adapter_sources as adapter_sources
        from polypbench.io import sha256
        root = Path(__file__).resolve().parents[1]
        cfg = {"project_root": str(root), "dataset": {"mask_format": "bitmask"}}
        helper = "polypbench/adapters/instance_lsj.py"
        for sources in (baseline_sources, scda_sources, adapter_sources):
            with self.subTest(runner=sources.__module__):
                self.assertEqual(sources(cfg)[helper], sha256(root / helper))


if __name__ == "__main__":
    unittest.main()
