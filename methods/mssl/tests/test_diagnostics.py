import unittest
import numpy as np

from methods.mssl.diagnostics import candidates, geometry


class DiagnosticTests(unittest.TestCase):
    def test_one_prediction_cannot_cover_two_assigned_gt(self):
        gt={"annotations":[{"id":1,"image_id":1,"bbox":[0,0,10,10]},
                           {"id":2,"image_id":1,"bbox":[1,0,10,10]},
                           {"id":3,"image_id":2,"bbox":[0,0,10,10]}]}
        actual=candidates(gt,[{"image_id":1,"bbox":[0,0,10,10],"score":.8}])
        self.assertEqual(sum(actual[i]["assigned_iou"]>0 for i in (1,2)),1)
        self.assertEqual(actual[3]["assigned_iou"],0.)
        self.assertGreater(actual[2]["best_iou_100"],.5)

    def test_geometry_preserves_gt_box_mismatch(self):
        mask=np.zeros((20,20),bool); mask[5:15,5:15]=True
        g=geometry(mask,[0,0,20,20])
        self.assertAlmostEqual(g["maskbox_iou"],.25)
        self.assertAlmostEqual(g["mask_inside_box_fraction"],.25)
        self.assertEqual(g["support_density_min"],0.)


if __name__=="__main__": unittest.main()
