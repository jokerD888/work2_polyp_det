import tempfile
import unittest
from pathlib import Path
import numpy as np

from polypbench.analysis.candidate_stage_jobs import full_cohort_sample
from polypbench.analysis.candidate_stage import aggregate_reports
from polypbench.io import read_json


class FullCohortTests(unittest.TestCase):
    def test_full_cohort_keeps_negatives_multigt_and_original_ids(self):
        gt = {"images": [{"id": 8, "height": 100, "width": 100, "center": "C2"},
                         {"id": 2, "height": 100, "width": 100, "center": "C1"},
                         {"id": 20, "height": 100, "width": 100, "center": "C3"}],
              "annotations": [{"id": 91, "image_id": 8, "bbox": [0,0,10,10]},
                              {"id": 92, "image_id": 8, "bbox": [20,20,20,20]},
                              {"id": 93, "image_id": 20, "bbox": [0,0,30,30]}]}
        cohort, cuts = full_cohort_sample(gt)
        self.assertEqual([im["id"] for im in cohort["images"]], [2,8,20])
        self.assertEqual([im["sampling_stratum"] for im in cohort["images"]], ["negative","multi","single"])
        self.assertEqual(sum(im["gt_count"] for im in cohort["images"]), 3)
        np.testing.assert_allclose(cuts, np.quantile([.01,.04,.09],[.25,.5,.75]))
        gt["images"].append(gt["images"][0])
        with self.assertRaises(ValueError):
            full_cohort_sample(gt)

    def test_aggregate_keeps_image_and_gt_denominators_separate(self):
        stage = lambda values, covered: {"max_iou": values, "best_query_or_token_id": [0]*len(values),
            "thresholds": {str(t): {"one_to_one_coverage": covered, "marginal_gt_covered": sum(v>=t for v in values)}
                           for t in (.5,.75,.9)}}
        positive = {"image_id": 1,"center":"C1","sampling_stratum":"multi","gt":2,
            "annotation_ids":[10,11],"stages":{"test":stage([1.,1.],1)},
            "gt_metadata":[{"relative_area_quartile":"q1"},{"relative_area_quartile":"q4"}],
            "native_top100_score_ge_05":3}
        negative = {"image_id":2,"center":"C1","sampling_stratum":"negative","gt":0,
            "annotation_ids":[],"stages":{"test":stage([],0)},"native_top100_score_ge_05":1}
        with tempfile.TemporaryDirectory() as tmp:
            aggregate_reports([positive,negative],Path(tmp),"full_polypgen")
            summary=read_json(Path(tmp)/"summary.json")
            self.assertTrue(summary["complete_benchmark"])
            self.assertEqual(summary["groups"]["all"]["images"],2)
            self.assertEqual(summary["groups"]["all"]["gt"],2)
            self.assertEqual(summary["groups"]["all"]["negative_images_with_score_ge_05"],1)
            metrics=summary["groups"]["all"]["stages"]["test"]["thresholds"]["0.5"]
            self.assertEqual(metrics["one_to_one_coverage"],1)
            self.assertEqual(metrics["marginal_gt_covered"],2)
            self.assertEqual(summary["groups"]["multiplicity=negative"]["gt"],0)


if __name__ == "__main__":
    unittest.main()
