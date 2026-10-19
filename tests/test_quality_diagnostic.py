import unittest

import numpy as np

from polypbench.analysis.quality_diagnostic import owners, pair_stats, same_gt_rows, assisted_scores, aggregate, fp_types
from polypbench.analysis.coco_cache import official_eval
from polypbench.analysis.matching import maximum_coverage


class QualityTests(unittest.TestCase):
    def test_multi_gt_ownership_excludes_ambiguous_and_background(self):
        matrix = np.array([[.9, .1, .7, .2], [.1, .8, .6, .1]])
        np.testing.assert_array_equal(owners(matrix), [0, 1, -1, -1])
        rows = same_gt_rows(matrix, np.array([.2, .3, .99, .98]), [11, 12])
        self.assertEqual([r['group_candidates'] for r in rows], [1, 1])
        self.assertEqual(sum(r['pairs'] for r in rows), 0)

    def test_pairs_have_meaningful_quality_gap_and_handle_score_ties(self):
        p = pair_stats([.9, .7, .71], [.3, .8, .3])
        self.assertEqual((p['pairs'], p['inversions'], p['score_ties']), (2, 1, 1))
        self.assertEqual(p['inversion_fraction'], .5)
        self.assertEqual(pair_stats([.8, .79], [.1, .9])['pairs'], 0)

    def test_highest_score_gap_and_threshold_crossings(self):
        rows = same_gt_rows(np.array([[.92, .7, .6]]), np.array([.3, .9, .2]), [11])
        self.assertAlmostEqual(rows[0]['top_score_iou_gap'], .22)
        self.assertTrue(rows[0]['t0.90_top_score_misses'])
        self.assertEqual(rows[0]['t0.90_cross_inversions'], 1)
        s = aggregate(rows)
        self.assertEqual(s['gt_top_score_gap_ge_005'], 1)

    def test_same_gt_permutation_keeps_background_and_other_gt_separate(self):
        matrix = np.array([[.9, .6, .1, .1, .7], [.1, .1, .95, .6, .7]])
        scores = np.array([.1, .8, .2, .7, .99])
        values = assisted_scores(matrix, scores)
        np.testing.assert_array_equal(values['same_gt_permutation'], [.8, .1, .7, .2, .99])
        np.testing.assert_array_equal(np.sort(values['image_gt_permutation']), np.sort(scores))
        np.testing.assert_array_equal(np.sort(values['same_gt_permutation']), np.sort(scores))

    def test_negative_frame_permutation_is_unchanged(self):
        scores = np.array([.8, .3, .1])
        values = assisted_scores(np.empty((0, 3)), scores)
        np.testing.assert_array_equal(values['image_gt_permutation'], scores)
        np.testing.assert_array_equal(values['same_gt_permutation'], scores)
        np.testing.assert_array_equal(values['direct_gt_iou'], [0., 0., 0.])

    def test_equal_quality_does_not_create_artificial_score_swaps(self):
        scores = np.array([.3, .8, .1])
        values = assisted_scores(np.array([[.8, .8, .8]]), scores)
        np.testing.assert_array_equal(values['image_gt_permutation'], scores)
        np.testing.assert_array_equal(values['same_gt_permutation'], scores)

    def test_selection_coverage_is_exclusive_not_per_gt_best_iou(self):
        # One candidate cannot cover both GT; the second resolves the conflict.
        matrix = np.array([[.9, .8], [.8, .1]])
        self.assertEqual(maximum_coverage(matrix[:, :1], .75), 1)
        self.assertEqual(maximum_coverage(matrix, .75), 2)

    def test_duplicate_detection_uses_official_matching(self):
        gt = {'images': [{'id': 1, 'width': 100, 'height': 100}],
              'categories': [{'id': 1, 'name': 'polyp'}],
              'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [10, 10, 20, 20], 'area': 400, 'iscrowd': 0}]}
        pred = [{'image_id': 1, 'category_id': 1, 'bbox': b, 'score': s}
                for b, s in [([10, 10, 20, 20], .9), ([10, 10, 20, 20], .8), ([70, 70, 10, 10], .7)]]
        counts = fp_types(official_eval(gt, pred), gt)['0.75']['10']
        self.assertEqual(counts['tp'], 1)
        self.assertEqual(counts['fp_duplicate_compatible'], 1)
        self.assertEqual(counts['fp_background'], 1)

    def test_missing_and_single_candidate_groups_keep_denominators(self):
        rows = same_gt_rows(np.array([[.8, .1], [.1, .2]]), np.array([.3, .4]), [1, 2])
        total = aggregate(rows)
        self.assertEqual(total['all_gt'], 2)
        self.assertEqual(total['gt_with_group'], 1)
        self.assertIsNone(total['micro_inversion_fraction'])


if __name__ == '__main__':
    unittest.main()
