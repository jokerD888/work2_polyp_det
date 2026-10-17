from dataclasses import replace
import math
import unittest
import torch
from scipy.ndimage import distance_transform_edt
import numpy as np

from methods.scda_proposal.scda_assigner import assign_scda, token_geometry, rectangle_coverage
from methods.scda_v3.config import V3Options
from methods.scda_v3.assigner import assign
from methods.scda_v3.mask_geometry import mask_sdf, sample_cell_centers, level_preference
from .fixtures import fixture


class AssignmentTests(unittest.TestCase):
    def test_v3a_token_gt_and_coordinate_scale_exactly_equal_legacy(self):
        for seed in range(5):
            torch.manual_seed(seed)
            payload, target = fixture()
            payload["raw_box_sigmoid"] = torch.rand_like(payload["raw_box_sigmoid"])
            for case in ("normal", "padding", "nonfinite", "empty", "unreliable"):
                if case == "padding": target.valid_pixels[:, 49:] = False
                if case == "nonfinite":
                    payload["raw_box_sigmoid"][0, 5] = float("nan")
                    payload["valid_token_mask"][0, 5] = False
                if case == "empty": target.masks.zero_()
                if case == "unreliable": target.reliable.zero_()
                options = V3Options()
                a = assign(payload, [target], options)[0]
                old = assign_scda(payload, [target], options.legacy())[0]
                self.assertTrue(torch.equal(a.reg.token_indices, old.reg.token_indices), (seed, case))
                self.assertTrue(torch.equal(a.reg.gt_indices, old.reg.gt_indices))
                self.assertTrue(torch.equal(a.box_scale, old.box_scale))
                self.assertEqual(a.cls.token_indices.numel(), 0)

    def test_non_square_cell_and_level_preference(self):
        order, scores = level_preference([(64, 64), (32, 32), (16, 16)], (256, 512), 8 * math.sqrt(8 * 16))
        self.assertEqual(order[0], 1)
        self.assertAlmostEqual(scores[1], 0)
        self.assertEqual(level_preference([(64, 64), (32, 32), (16, 16)], (256, 256), 32)[0][0], 0)
        self.assertEqual(level_preference([(64, 64), (32, 32), (16, 16)], (256, 256), 128)[0][0], 2)

    def test_sdf_sign_zero_padding_and_visible_cut_boundary(self):
        mask = torch.zeros(16, 20, dtype=torch.bool); mask[2:14, 2:18] = True
        valid = torch.ones_like(mask); valid[:, 13:] = False
        sdf, diameter = mask_sdf(mask, valid)
        self.assertGreater(float(sdf[7, 7]), 0)
        self.assertLess(float(sdf[7, 15]), 0)
        self.assertGreater(float(sdf[7, 10]), float(sdf[7, 12]))
        visible = (mask & valid).numpy()
        expected = distance_transform_edt(np.pad(visible, 1)) - distance_transform_edt(~np.pad(visible, 1))
        torch.testing.assert_close(sdf, torch.tensor(expected[1:-1, 1:-1] / math.sqrt(visible.sum()/math.pi)))
        self.assertAlmostEqual(diameter, 2 * math.sqrt(visible.sum()/math.pi))

    def test_sdf_full_canvas_has_finite_border_and_empty_rejected(self):
        full = torch.ones(5, 7, dtype=torch.bool)
        sdf, _ = mask_sdf(full, full)
        self.assertTrue(torch.isfinite(sdf).all())
        self.assertGreater(float(sdf[2, 3]), float(sdf[0, 0]))
        with self.assertRaises(ValueError): mask_sdf(full, ~full)
        with self.assertRaises(ValueError): mask_sdf(full.float()*float("nan"), full)

    def test_sdf_bilinear_sampling_actual_pixel_centers(self):
        field = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
        cells = torch.tensor([[0., 0., 1., 1.], [0., 0., 2., 2.]], dtype=torch.float64)
        torch.testing.assert_close(sample_cell_centers(field, cells), torch.tensor([1., 2.5], dtype=torch.float64))

    def test_v3b_mask_contour_changes_selection_even_with_same_coverage_pool_and_area(self):
        payload, target = fixture(shapes=((8, 8),))
        target.boxes[:] = torch.tensor([.5, .5, 1., 1.])
        target.masks[:] = True
        # Every cell loses exactly 4 pixels; positions alter SDF at cell centers
        # while per-cell coverage, mask area, boxes and native predictions stay fixed.
        first, second = target.masks.clone(), target.masks.clone()
        for y in range(8):
            for x in range(8):
                offset = 3 if (x + y) % 2 else 0
                first[:, y*8+offset:y*8+offset+2, x*8+offset:x*8+offset+2] = False
                other = 0 if offset else 3
                second[:, y*8+other:y*8+other+2, x*8+other:x*8+other+2] = False
        cells, _, _ = token_geometry(payload, 0, target)
        self.assertTrue(torch.equal(rectangle_coverage(first, cells), rectangle_coverage(second, cells)))
        torch.manual_seed(12)
        payload["raw_box_sigmoid"] = torch.rand_like(payload["raw_box_sigmoid"])
        target.masks = first
        a = assign(payload, [target], V3Options(assignment_policy="mask_multiscale"))[0]
        target.masks = second
        b = assign(payload, [target], V3Options(assignment_policy="mask_multiscale"))[0]
        self.assertEqual(a.audit["per_gt"][0]["pool_count"], b.audit["per_gt"][0]["pool_count"])
        self.assertNotEqual(a.reg.token_indices.tolist(), b.reg.token_indices.tolist())

    def test_v3b_padding_topq_caps_regions_and_determinism(self):
        payload, target = fixture()
        target.valid_pixels[:, 49:] = False
        options = V3Options(assignment_policy="mask_multiscale")
        a, b = (assign(payload, [target], options)[0] for _ in range(2))
        self.assertTrue(torch.equal(a.reg.token_indices, b.reg.token_indices))
        p = a.reg.token_indices.tolist()
        self.assertEqual(len(p), len(set(p)))
        self.assertLessEqual(len(p), 4)
        self.assertFalse(set(p) & {0, 1})
        cells, _, _ = token_geometry(payload, 0, target)
        self.assertTrue((cells[a.reg.token_indices, 2] <= 49).all())
        self.assertLessEqual(len(a.audit["per_gt"][0]["preferred_levels"]), 2)

    def test_empty_multibox_small_and_nonfinite_candidate_cases(self):
        options = V3Options(assignment_policy="mask_multiscale")
        for case in ("empty_gt", "multibox", "empty_mask", "tiny", "nonfinite"):
            payload, target = fixture(gt_count=0 if case == "empty_gt" else (2 if case == "multibox" else 1))
            if case == "multibox": target.reliable.zero_()
            if case == "empty_mask": target.masks.zero_()
            if case == "tiny": target.boxes[:] = torch.tensor([.01, .01, .001, .001])
            if case == "nonfinite": payload["valid_token_mask"].zero_()
            self.assertEqual(assign(payload, [target], options)[0].reg.token_indices.numel(), 0, case)

    def test_multibox_cannot_become_reliable_after_crop(self):
        payload, target = fixture()
        target.original_gt_count = 2
        for policy in ("legacy_quality_iou", "mask_multiscale"):
            with self.assertRaisesRegex(ValueError, "ORIGINAL single"):
                assign(payload, [target], V3Options(assignment_policy=policy))

    def test_v3b_refills_and_shortfalls_are_explicit(self):
        payload, target = fixture(shapes=((8, 8),))
        result = assign(payload, [target], V3Options(assignment_policy="mask_multiscale"))[0]
        entry = result.audit["per_gt"][0]
        self.assertEqual(len(entry["selected"]), 4)
        self.assertEqual(entry["same_level_refill"], 2)
        payload["valid_token_mask"].zero_(); payload["valid_token_mask"][0, 27] = True
        result = assign(payload, [target], V3Options(assignment_policy="mask_multiscale"))[0]
        entry = result.audit["per_gt"][0]
        self.assertEqual(entry["regions"], ["boundary"])
        self.assertEqual(entry["shortfall"], 3)


if __name__ == "__main__": unittest.main()
