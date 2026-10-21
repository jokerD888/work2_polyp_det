import unittest
import numpy as np
from polypbench.analysis.initialization_identifiability import (
    FEATURES,before_decoder_features,ownership,outcome,rank_metrics,balanced_weights,subset_report)


class IdentifiabilityTests(unittest.TestCase):
    def pre(self):
        boxes=np.tile([.5,.5,.5,.5],(300,1))
        return {'init_encoder_normalized':boxes.copy(),'init_mask_normalized':boxes.copy(),
            'mask2box_raw_lowres_xyxy':np.tile([0,0,1,1],(300,1)),
            'mask_foreground_pixels':np.full(300,4),'mask_foreground_fraction':np.full(300,.25),
            'mask_padding_foreground_fraction':np.zeros(300),'mask_empty':np.zeros(300,bool),
            'mask_one_row':np.zeros(300,bool),'mask_one_column':np.zeros(300,bool)}

    def test_features_use_no_GT_or_final_predictions_and_rank_is_preselection_order(self):
        pre=self.pre(); a=before_decoder_features(pre)
        pre.update(gt_xyxy=np.random.default_rng(42).random((5,4)),final_class_scores=np.ones(300),
            decoder_09_gt_iou=np.ones((5,300)),final_top100_mask_scores=np.ones(100))
        b=before_decoder_features(pre)
        np.testing.assert_equal(a,b)
        np.testing.assert_equal(a[:,FEATURES.index('encoder_selection_rank')],np.arange(1,301))
        self.assertTrue(np.isnan(a[:,FEATURES.index('encoder_low_selection_score')]).all())
        self.assertTrue(np.isnan(a[:,FEATURES.index('encoder_not_native_legal')]).all())
        self.assertTrue((a[:,FEATURES.index('mask_bbox_unfilled_fraction')]==0).all())

    def test_empty_and_boundary_references_are_retained_with_explicit_missing_occupancy(self):
        pre=self.pre(); pre['mask_empty'][0]=True
        pre['mask_foreground_pixels'][0]=0
        pre['mask2box_raw_lowres_xyxy'][0]=[1e8,1e8,0,0]
        pre['init_encoder_normalized'][0]=[1,1,1,1]
        features=before_decoder_features(pre)
        self.assertEqual(len(features),300)
        self.assertTrue(np.isnan(features[0,FEATURES.index('mask_bbox_unfilled_fraction')]))
        self.assertEqual(features[0,FEATURES.index('encoder_boundary')],1)

    def test_ownership_retains_ambiguity_switches_and_negative_queries(self):
        mats=[np.zeros((2,300)) for _ in range(5)]
        mats[0][0,0]=.7; mats[1][1,0]=.8
        mats[2][0,1]=.7; mats[3][1,1]=.8
        mats[2][0,2]=.9; mats[3][0,2]=.9
        owner,initial,final=ownership(mats)
        self.assertEqual(owner[0],-2); self.assertTrue(initial[0])
        self.assertEqual(owner[1],-2); self.assertTrue(final[1])
        self.assertEqual(owner[2],0); self.assertEqual(owner[3],-1)
        negative=ownership([np.empty((0,300)) for _ in range(5)])
        self.assertTrue((negative[0]==-1).all())

    def test_all_four_outcomes_and_prevalence_are_kept(self):
        labels=outcome([.1,.1,.9,.9],[.1,.9,.1,.9],.5)
        np.testing.assert_equal(labels,[0,1,2,3])
        x=np.zeros((4,len(FEATURES)))
        r=subset_report(x,labels,np.array([0,0,1,1]),np.ones(4,bool))
        self.assertEqual(r['counts'],dict(both_bad=1,recovered=1,damaged=1,both_good=1))
        self.assertEqual(r['features'][FEATURES[0]]['native_covered_control']['queries'],2)

    def test_image_weights_prevent_duplicate_queries_becoming_independent_images(self):
        images=np.array([0,0,0,1])
        w=balanced_weights(images,np.ones(4,bool))
        self.assertAlmostEqual(w[:3].sum(),1); self.assertEqual(w[3],1)

    def test_rank_ties_and_base_rate_and_no_automatic_score_flipping(self):
        tied=rank_metrics(np.ones(10),np.arange(10)==0,np.ones(10))
        self.assertAlmostEqual(tied['auc'],.5); self.assertAlmostEqual(tied['ranking_ap'],.1)
        perfect=rank_metrics([0,1,2,3],[0,0,1,1],np.ones(4))
        self.assertEqual(perfect['auc'],1); self.assertEqual(perfect['ranking_ap'],1)
        reverse=rank_metrics([3,2,1,0],[0,0,1,1],np.ones(4))
        self.assertEqual(reverse['auc'],0)
        self.assertIsNone(rank_metrics([0,1],[0,0],np.ones(2))['auc'])

    def test_weighted_auc_and_grouped_average_precision(self):
        # Two positive units tied with three negatives; the high-scoring positive
        # has weight four. Weighted pair counting gives 19/24 concordance.
        metric=rank_metrics([1,1,2,3],[1,0,0,1],np.array([2.,3.,1.,4.]))
        self.assertAlmostEqual(metric['auc'],19/24)
        self.assertAlmostEqual(metric['ranking_ap'],4/6+(2/6)*.6)
        self.assertAlmostEqual(metric['prevalence'],.6)

    def test_redundant_query_recovery_is_not_new_GT_coverage(self):
        from polypbench.analysis.initialization_identifiability_report import useful_witnesses
        result=useful_witnesses(np.array([.1,.1,.1,.1]),np.array([.8,.8,.8,.8]),
            np.array([0,1,-1,-2]),np.array([.8,.2]),np.array([.9,.8]),.5)
        np.testing.assert_equal(result,[False,True,False,False])


if __name__=='__main__': unittest.main()
