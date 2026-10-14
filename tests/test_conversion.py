import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_api

from polypbench.data import validate_coco, verify_supplied_boxes
from polypbench.io import read_json, write_json
from polypbench.kvasir import convert_kvasir


class ConversionTests(unittest.TestCase):
    def make_source(self, root, count=10):
        source = root / "source"
        (source / "images").mkdir(parents=True)
        (source / "masks").mkdir()
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[3:10, 3:10] = 255
        for index in range(count):
            Image.new("RGB", (16, 16), "red").save(source / "images" / f"{index}.png")
            Image.fromarray(mask).save(source / "masks" / f"{index}.png")
        return source

    def test_official_split_order_is_preserved_without_test_or_shuffle(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.make_source(root)
            split_dir = root / "split"
            split_dir.mkdir()
            (split_dir / "train.txt").write_text("8\n6\n4\n2\n0\n1\n3\n5\n", encoding="utf-8")
            (split_dir / "val.txt").write_text("9.png\n7.png\n", encoding="utf-8")
            first = convert_kvasir(source, root / "first", seed=42, split_dir=split_dir, bbox_source="mask-components")
            second = convert_kvasir(source, root / "second", seed=99, split_dir=split_dir, bbox_source="mask-components")
            self.assertEqual(first["splits"], second["splits"])
            self.assertEqual(first["splits"]["val"]["files"], ["9.png", "7.png"])
            self.assertNotIn("test", first["splits"])
            self.assertFalse((root / "first" / "test.json").exists())
            self.assertEqual((root / "first" / "train.txt").read_bytes(), (split_dir / "train.txt").read_bytes())
            self.assertFalse(first["seed_used_for_split"])

    def test_bad_official_split_is_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.make_source(root)
            split_dir = root / "split"
            split_dir.mkdir()
            (split_dir / "train.txt").write_text("\n".join(str(i) for i in range(8)))
            for invalid, message in (("7\n9", "overlapping"), ("9\nmissing", "missing image"), ("9", "do not cover")):
                (split_dir / "val.txt").write_text(invalid)
                with self.assertRaisesRegex(ValueError, message):
                    convert_kvasir(source, root / "output", split_dir=split_dir, bbox_source="mask-components")
                self.assertFalse((root / "output").exists())

    def test_holes_components_and_split_reproducibility(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            (source / "images").mkdir(parents=True)
            (source / "masks").mkdir()
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[3:20, 4:21] = 255
            mask[8:12, 9:13] = 0  # hole must survive conversion
            mask[25:29, 25:29] = 255  # second instance
            mask[0, 0] = 255  # explicit sub-5-pixel filtering
            for index in range(10):
                Image.new("RGB", (32, 32), "red").save(source / "images" / f"{index}.png")
                Image.fromarray(mask).save(source / "masks" / f"{index}.png")
            first = convert_kvasir(source, root / "first", bbox_source="mask-components")
            second = convert_kvasir(source, root / "second", bbox_source="mask-components")
            self.assertEqual(first["splits"], second["splits"])
            self.assertEqual(len(first["discarded_components"]), 10)
            for split, count in (("train", 8), ("val", 1), ("test", 1)):
                data = read_json(root / "first" / f"{split}.json")
                self.assertEqual(len(data["images"]), count)
                self.assertEqual(len(data["annotations"]), count * 2)
                for image in data["images"]:
                    decoded = np.zeros((32, 32), dtype=bool)
                    for ann in data["annotations"]:
                        if ann["image_id"] == image["id"]:
                            decoded |= mask_api.decode(ann["segmentation"]).astype(bool)
                    expected = mask > 127
                    expected[0, 0] = False
                    np.testing.assert_array_equal(decoded, expected)
                validate_coco(root / "first" / f"{split}.json", root / "first" / split, require_masks=True, mask_format="bitmask")
            all_files = [f for split in first["splits"].values() for f in split["files"]]
            self.assertEqual(len(all_files), len(set(all_files)))
            self.assertTrue((source / "images" / "0.png").is_file())

    def test_supplied_boxes_are_copied_exactly_including_tiny_annotations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.make_source(root)
            # Box extents intentionally differ from mask extents; tiny supplied
            # annotations must not be removed by the mask-components area filter.
            boxes = [{"label": "polyp", "xmin": 1, "ymin": 1, "xmax": 14, "ymax": 14}, {"label": "polyp", "xmin": 3, "ymin": 3, "xmax": 4, "ymax": 4}]
            supplied = {str(index): {"width": 16, "height": 16, "bbox": boxes} for index in range(10)}
            source_json = source / "kavsir_bboxes.json"
            write_json(source_json, supplied)
            original = source_json.read_bytes()
            audit = convert_kvasir(source, root / "output", min_area=1000)
            self.assertEqual(audit["bbox_source_kind"], "provided")
            self.assertEqual(audit["discarded_components"], [])
            self.assertEqual((root / "output" / audit["bbox_source_file"]).read_bytes(), original)
            for split in audit["splits"]:
                data = read_json(root / "output" / f"{split}.json")
                verify_supplied_boxes(root / "output" / f"{split}.json", source_json)
                for image in data["images"]:
                    annotations = [ann for ann in data["annotations"] if ann["image_id"] == image["id"]]
                    self.assertEqual([ann["bbox"] for ann in annotations], [[1, 1, 13, 13], [3, 3, 1, 1]])
                    self.assertEqual([ann["area"] for ann in annotations], [169, 1])
                    self.assertEqual([ann["source_bbox_index"] for ann in annotations], [0, 1])
                    self.assertEqual(int(mask_api.decode(annotations[1]["segmentation"]).sum()), 1)
            self.assertEqual(source_json.read_bytes(), original)
            tampered = root / "output" / "val.json"
            data = read_json(tampered)
            data["annotations"][0]["bbox"][0] += 1
            write_json(tampered, data)
            with self.assertRaisesRegex(ValueError, "coordinates changed"):
                verify_supplied_boxes(tampered, source_json)

    def test_empty_auxiliary_mask_is_reported_without_fabricating_a_box_mask(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.make_source(root)
            supplied = {str(index): {"width": 16, "height": 16, "bbox": [{"label": "polyp", "xmin": 0, "ymin": 0, "xmax": 2, "ymax": 2}]} for index in range(10)}
            write_json(source / "kavsir_bboxes.json", supplied)
            with self.assertRaisesRegex(ValueError, "has no mask foreground"):
                convert_kvasir(source, root / "output")


if __name__ == "__main__":
    unittest.main()
