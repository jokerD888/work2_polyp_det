import tempfile
import unittest
from pathlib import Path
import numpy as np
from polypbench.analysis.uncertainty import run
from polypbench.io import sha256


class ReusedBootstrapTests(unittest.TestCase):
    def test_exact_old_draws_and_new_competitor_are_supported(self):
        gt = {"images": [{"id": i, "width": 20, "height": 20, "center": "C1" if i < 3 else "C2"} for i in (1, 2, 3)],
              "categories": [{"id": 1, "name": "polyp"}],
              "annotations": [{"id": i, "image_id": i, "category_id": 1, "bbox": [1, 1, 10, 10],
                               "area": 100, "iscrowd": 0} for i in (1, 2, 3)]}
        rows = [{"image_id": i, "category_id": 1, "bbox": [1, 1, 10, 10], "score": .9} for i in (1, 2, 3)]
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); draws = np.array([[2, 0, 1], [0, 2, 1], [1, 1, 1]], dtype=np.uint16)
            np.save(base/"original.npy", draws)
            result = run(gt, {"baseline": rows, "mssl": rows, "ciou_replace": rows}, base/"report",
                         replicates=3, workers=1, draws_path=base/"original.npy")
            self.assertEqual(sha256(base/"original.npy"), sha256(base/"report/image_multiplicities.npy"))
            value = result["paired_intervals"]["ciou_replace"]["pooled"]["AP"]
            self.assertEqual(value["point_delta_pp"], 0.)
            self.assertEqual(value["low_pp"], 0.)
            self.assertEqual(value["high_pp"], 0.)
            np.save(base/"bad.npy", np.array([[1, 1, 0]]*3, dtype=np.uint16))
            with self.assertRaisesRegex(ValueError, "composition"):
                run(gt, {"mssl": rows}, base/"invalid", replicates=3, workers=1, draws_path=base/"bad.npy")
