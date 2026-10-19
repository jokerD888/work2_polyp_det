import unittest
import numpy as np

from polypbench.analysis.coco_cache import CocoCache,duplicate_dataset,official_eval,ap_vector
from polypbench.analysis.matching import maximum_coverage


class AnalysisTests(unittest.TestCase):
    def test_threshold_matching_resolves_greedy_counterexample(self):
        matrix=np.array([[.95,.81],[.85,.1]])
        self.assertEqual(maximum_coverage(matrix,.8),2)
        self.assertEqual(maximum_coverage(matrix,.8,1),1)
        self.assertEqual(maximum_coverage(np.empty((0,4)),.8),0)
        # Max total IoU would choose .99+.79, yielding only one valid match.
        self.assertEqual(maximum_coverage(np.array([[.99,.81],[.81,.79]]),.8),2)

    def fixture(self):
        gt={"images":[{"id":i,"width":100,"height":100} for i in (1,2,3)],
            "categories":[{"id":1,"name":"polyp"}],
            "annotations":[{"id":1,"image_id":1,"category_id":1,"bbox":[10,10,20,20],"area":400,"iscrowd":0},
                           {"id":2,"image_id":2,"category_id":1,"bbox":[20,20,20,20],"area":400,"iscrowd":0}]}
        pred=[{"image_id":1,"category_id":1,"bbox":[10,10,20,20],"score":.8},
              {"image_id":1,"category_id":1,"bbox":[70,70,10,10],"score":.8},
              {"image_id":2,"category_id":1,"bbox":[20,20,20,20],"score":.8},
              {"image_id":3,"category_id":1,"bbox":[0,0,20,20],"score":.9}]
        return gt,pred

    def test_cached_bootstrap_equals_fresh_coco_with_copies_and_score_ties(self):
        gt,pred=self.fixture(); cache=CocoCache(gt,pred)
        for sampled in ([1,1,2,3,3],[1,1,1],[1,2,2,2,3],[2,3,3]):
            clone,rows=duplicate_dataset(gt,pred,sampled)
            self.assertEqual(len({i["id"] for i in clone["images"]}),len(sampled))
            self.assertEqual(len({a["id"] for a in clone["annotations"]}),len(clone["annotations"]))
            counts=np.array([sampled.count(i) for i in cache.ids])
            np.testing.assert_allclose(cache.accumulate(counts),ap_vector(official_eval(clone,rows)),rtol=0,atol=1e-14)

    def test_negative_frames_remain_in_bootstrap_and_change_precision(self):
        gt,pred=self.fixture(); cache=CocoCache(gt,pred)
        self.assertGreater(cache.accumulate([1,1,0]).mean(),cache.accumulate([1,1,4]).mean())
        self.assertTrue(np.isnan(cache.accumulate([0,0,3])).all())

    def test_restricted_center_cache_equals_full_resample(self):
        gt,pred=self.fixture(); cache=CocoCache(gt,pred); subset=cache.restrict([True,False,True])
        np.testing.assert_allclose(subset.accumulate([3,2]),cache.accumulate([3,0,2]),rtol=0,atol=1e-14)


if __name__=="__main__": unittest.main()
