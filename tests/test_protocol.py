import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from polypbench.config import load_config
from polypbench.data import validate_coco, validate_dataset
from polypbench.evaluation import evaluate
from polypbench.io import read_json, write_json
from polypbench.toy import create_toy


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "toy"
        create_toy(self.data)
        self.gt = self.data / "val.json"
        self.pred = self.root / "predictions.json"

    def tearDown(self):
        self.temp.cleanup()

    def evaluate(self, predictions):
        write_json(self.pred, predictions)
        with contextlib.redirect_stdout(io.StringIO()):
            return evaluate(str(self.gt), str(self.pred))

    def predictions(self):
        return [{"image_id": ann["image_id"], "category_id": ann["category_id"], "bbox": ann["bbox"], "score": .9} for ann in read_json(self.gt)["annotations"]]

    def test_perfect_predictions_with_noncontiguous_category_ids(self):
        result = self.evaluate(self.predictions())
        self.assertAlmostEqual(result["metrics"]["AP"], 1)
        self.assertEqual(result["protocol"]["max_dets"], [1, 10, 100])

    def test_empty_predictions_are_zero_not_exception(self):
        result = self.evaluate([])
        self.assertEqual(result["metrics"]["AP"], 0)
        self.assertIsNone(result["metrics"]["AP_small"])

    def test_unpredicted_images_still_count(self):
        result = self.evaluate(self.predictions()[:1])
        self.assertEqual(result["images"], 2)
        self.assertAlmostEqual(result["metrics"]["AR100"], .5)
        self.assertLess(result["metrics"]["AP"], .6)

    def test_unknown_category_rejected(self):
        rows = self.predictions()
        rows[0]["category_id"] = 0
        with self.assertRaisesRegex(ValueError, "ID not in ground truth"):
            self.evaluate(rows)

    def test_no_high_score_filter(self):
        rows = self.predictions()
        for row in rows:
            row["score"] = .00001
        self.assertAlmostEqual(self.evaluate(rows)["metrics"]["AP"], 1)

    def test_bbox_only_and_rle_rejected_for_native_training(self):
        for segmentation in (None, {"size": [256, 256], "counts": "encoded"}):
            data = read_json(self.gt)
            data["annotations"][0]["segmentation"] = segmentation
            target = self.root / "invalid.json"
            write_json(target, data)
            with self.assertRaisesRegex(ValueError, "requires polygon"):
                validate_coco(target, require_masks=True, check_images=False)

    def test_duplicate_ids_and_wrong_image_dimensions_rejected(self):
        data = read_json(self.gt)
        data["images"][1]["id"] = data["images"][0]["id"]
        target = self.root / "invalid.json"
        write_json(target, data)
        with self.assertRaisesRegex(ValueError, "duplicate image"):
            validate_coco(target, check_images=False)
        data = read_json(self.gt)
        data["images"][0]["width"] = 255
        write_json(target, data)
        with self.assertRaisesRegex(ValueError, "Image size differs"):
            validate_coco(target, self.data / "val")

    def test_identical_images_across_splits_rejected(self):
        cfg = {"model": {"num_classes": 1}, "dataset": {"train": {"annotations": str(self.gt), "image_root": str(self.data / "val")}, "val": {"annotations": str(self.gt), "image_root": str(self.data / "val")}}}
        with self.assertRaisesRegex(ValueError, "Image shared across"):
            validate_dataset(cfg)

    def test_epoch_budget_matches_batch_and_image_count(self):
        cfg = {"model": {"num_classes": 1}, "dataset": {"train": {"annotations": str(self.data / "train.json"), "image_root": str(self.data / "train")}, "val": {"annotations": str(self.gt), "image_root": str(self.data / "val")}}, "overrides": {"SOLVER.IMS_PER_BATCH": 2, "SOLVER.MAX_ITER": 20}, "training_budget": {"train_images": 4, "iterations_per_epoch": 2, "epochs": 10}}
        validate_dataset(cfg)
        cfg["overrides"]["SOLVER.IMS_PER_BATCH"] = 1
        with self.assertRaisesRegex(ValueError, "does not match train image"):
            validate_dataset(cfg)
        cfg["overrides"]["SOLVER.IMS_PER_BATCH"] = 2
        cfg["overrides"]["SOLVER.MAX_ITER"] = 21
        with self.assertRaisesRegex(ValueError, "does not match the declared epoch"):
            validate_dataset(cfg)


if __name__ == "__main__":
    unittest.main()
