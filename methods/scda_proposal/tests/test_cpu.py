"""Geometry/assignment/loss tests that never initialize a CUDA context."""
import dataclasses
import math
import tempfile
from pathlib import Path
import socket
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from methods.scda_proposal.config import SCDAOptions
from methods.scda_proposal.scda_assigner import assign_scda, rectangle_coverage, fully_valid_cells, token_geometry
from methods.scda_proposal.scda_loss import scda_losses, connected_zero
from methods.scda_proposal.structures import SCDATarget, TaskAssignment, ImageAssignment

torch.set_num_threads(2)


def fixture(gt_count=1, reliable=True, shape=(8, 8), grid=(4, 4), image_size=None):
    h, w = shape
    gh, gw = grid
    y, x = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
    anchors = torch.stack(((x + .5) / gw, (y + .5) / gh,
                           torch.full_like(x, .2, dtype=torch.float),
                           torch.full_like(y, .2, dtype=torch.float)), -1).reshape(1, -1, 4)
    n = gh * gw
    payload = {"raw_cls": torch.zeros(1, n, 2, requires_grad=True),
               "raw_box_sigmoid": anchors.clone().requires_grad_(), "base_anchor": anchors,
               "topk_indices": torch.tensor([[0]]), "spatial_shapes": torch.tensor([[gh, gw]]),
               "level_start_index": torch.tensor([0]), "mask_flatten": torch.zeros(1, n, dtype=torch.bool),
               "valid_token_mask": torch.ones(1, n, dtype=torch.bool)}
    target = SCDATarget(torch.zeros(gt_count, dtype=torch.long),
                        torch.tensor([[.5, .5, .75, .75]]).repeat(gt_count, 1),
                        torch.ones(gt_count, h, w, dtype=torch.bool),
                        torch.full((gt_count,), reliable, dtype=torch.bool),
                        torch.arange(gt_count), torch.ones(h, w, dtype=torch.bool),
                        image_size or shape, original_gt_count=gt_count)
    return payload, target


def rank_case(model, rank, empty_global=False):
    value = torch.arange(1, 5, dtype=torch.float).view(1, 4, 1) / 4
    out = model(value)
    payload = {"raw_cls": out[..., :1], "raw_box_sigmoid": out[..., 1:].sigmoid()}
    g = 0 if rank == 1 or empty_global else 2
    target = SCDATarget(torch.zeros(g, dtype=torch.long), torch.tensor([[.5, .4, .35, .3]]).repeat(g, 1),
                        torch.zeros(g, 1, 1), torch.ones(g, dtype=torch.bool), torch.arange(g),
                        torch.ones(1, 1, dtype=torch.bool), (1, 1))
    p, j = (torch.tensor([0, 1, 2]), torch.tensor([0, 0, 1])) if g else (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
    assigned = ImageAssignment(TaskAssignment(p, j), TaskAssignment(p, j), torch.ones(4, 4))
    return payload, target, assigned


def gloo_rank(rank, port, output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    try:
        model = torch.nn.Linear(1, 5, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[.2], [.3], [.4], [.5], [.6]]))
        ddp = DistributedDataParallel(model)
        records = {}
        for empty in (False, True):
            ddp.zero_grad(set_to_none=True)
            payload, target, assigned = rank_case(ddp, rank, empty)
            result = scda_losses(payload, [target], [assigned], SCDAOptions(), 5, 2)
            sum(result.losses.values()).backward()
            records[str(empty)] = {"grad": model.weight.grad.clone(), "counts": result.global_gt_counts}
        torch.save(records, Path(output) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


class GeometryTests(unittest.TestCase):
    def test_fractional_integral_matches_independent_pixel_intersection(self):
        mask = torch.tensor([[[1, 0, 1], [0, 1, 1]]], dtype=torch.bool)
        cells = torch.tensor([[.25, .3, 2.7, 1.8], [0, 0, 1, 1], [.9, .8, 1.1, 1.2]], dtype=torch.float64)
        observed = rectangle_coverage(mask, cells)[0]
        expected = []
        for x0, y0, x1, y1 in cells.tolist():
            integral = 0
            for y in range(2):
                for x in range(3):
                    if mask[0, y, x]:
                        integral += max(min(x1, x+1)-max(x0, x), 0) * max(min(y1, y+1)-max(y0, y), 0)
            expected.append(integral / ((x1-x0)*(y1-y0)))
        torch.testing.assert_close(observed, torch.tensor(expected, dtype=torch.float64), rtol=1e-12, atol=1e-12)

    def test_partial_padding_discarded(self):
        valid = torch.tensor([[True, True, False], [True, True, False]])
        cells = torch.tensor([[0, 0, 2, 2], [1.5, 0, 2.5, 1], [2, 0, 3, 1]], dtype=torch.float64)
        self.assertEqual(fully_valid_cells(valid, cells).tolist(), [True, False, False])

    def test_different_canvas_and_image_normalization(self):
        payload, target = fixture(shape=(8, 12), grid=(4, 6), image_size=(6, 9))
        cells, scales, _ = token_geometry(payload, 0, target)
        torch.testing.assert_close(cells[0], torch.tensor([0, 0, 2, 2], dtype=torch.float64))
        torch.testing.assert_close(scales[0], torch.tensor([4/3]*4))


class AssignmentTests(unittest.TestCase):
    def setUp(self):
        self.options = SCDAOptions(enabled=True, diagnostics=True)

    def test_caps_subset_topq_and_determinism(self):
        payload, target = fixture()
        first = assign_scda(payload, [target], self.options)[0]
        second = assign_scda(payload, [target], self.options)[0]
        cls, reg = first.cls.token_indices.tolist(), first.reg.token_indices.tolist()
        self.assertEqual(len(cls), 2)
        self.assertEqual(len(reg), 4)
        self.assertTrue(set(cls) <= set(reg))
        self.assertNotIn(0, reg)
        self.assertEqual(reg, second.reg.token_indices.tolist())
        entry = first.audit["per_gt"][0]
        self.assertEqual(entry["cls_reg_overlap_fraction"], 1)
        self.assertEqual(entry["reg_only_fraction"], .5)
        self.assertFalse(entry["reg_threshold_changes_selection"])

    def test_transition_threshold_really_changes_selection_when_core_insufficient(self):
        payload, target = fixture()
        target.masks[:] = False
        target.masks[:, 2:4, 2:4] = True  # one full cell
        target.masks[:, 2:3, 4:6] = True  # adjacent half-covered cell
        selected = assign_scda(payload, [target], self.options)[0]
        entry = selected.audit["per_gt"][0]
        self.assertGreater(entry["reg_transition_count"], 0)
        self.assertTrue(entry["reg_threshold_changes_selection"])
        self.assertTrue(entry["cls_subset_reg"])

    def test_empty_unreliable_empty_mask_small_thin_border(self):
        for case in ("no_gt", "unreliable", "empty_mask", "small", "thin", "border", "invalid_box"):
            payload, target = fixture(gt_count=0 if case == "no_gt" else 1, reliable=case != "unreliable")
            if case == "empty_mask": target.masks[:] = False
            if case == "small": target.boxes[:] = torch.tensor([.01, .01, .001, .001])
            if case == "thin": target.boxes[:] = torch.tensor([.5, .5, .01, .7])
            if case == "border": target.boxes[:] = torch.tensor([.01, .01, .01, .01])
            if case == "invalid_box": target.boxes[:, 2] = 0
            result = assign_scda(payload, [target], self.options)[0]
            self.assertEqual(result.cls.token_indices.numel(), 0, case)
            self.assertEqual(result.reg.token_indices.numel(), 0, case)

    def test_shared_owner_for_overlapping_gt(self):
        payload, target = fixture(gt_count=2)
        target.gt_ids = torch.tensor([20, 10])
        assigned = assign_scda(payload, [target], self.options)[0]
        self.assertTrue((assigned.cls.gt_indices == 1).all())
        self.assertTrue((assigned.reg.gt_indices == 1).all())
        self.assertGreater(assigned.audit["n_conflicting_tokens"], 0)

    def test_invalid_and_padding_tokens_excluded(self):
        payload, target = fixture()
        payload["valid_token_mask"][:, 5] = False
        target.valid_pixels[2:4, 4:6] = False
        assigned = assign_scda(payload, [target], self.options)[0]
        self.assertNotIn(5, assigned.reg.token_indices.tolist())
        self.assertNotIn(6, assigned.reg.token_indices.tolist())


class V2AssignmentTests(unittest.TestCase):
    def setUp(self):
        self.options = SCDAOptions(enabled=True, diagnostics=True, reg_selection="core_plus_transition")

    def case(self, values):
        payload, target = fixture()
        target.boxes[:] = torch.tensor([.5, .5, 1., 1.])
        target.masks[:] = False
        for token, value in enumerate(values):
            y, x = divmod(token, 4)
            for pixel in range(round(value * 4)):
                dy, dx = divmod(pixel, 2)
                target.masks[0, y*2+dy, x*2+dx] = True
        return payload, target, assign_scda(payload, [target], self.options)[0]

    def test_two_core_two_transition_despite_surplus_core(self):
        values = [1.] * 12 + [.75, .5, .25, 0.]
        payload, target, assigned = self.case(values)
        entry = assigned.audit["per_gt"][0]
        self.assertEqual(entry["n_reg_core_selected"], 2)
        self.assertEqual(entry["n_reg_transition_selected"], 2)
        self.assertEqual(entry["reg_transition_shortfall"], 0)
        self.assertEqual(entry["reg_only_fraction"], .5)
        self.assertTrue(entry["reg_threshold_changes_selection"])
        self.assertEqual(assigned.reg.token_indices[:2].tolist(), assigned.cls.token_indices.tolist())
        self.assertEqual(assigned.reg.token_indices[2:].tolist(), [13, 14])
        self.assertNotIn(0, assigned.reg.token_indices.tolist())
        # Exact boundaries: .75 is core and .25 is medium.
        self.assertEqual(entry["n_cls_owned"], 12)
        legacy = assign_scda(payload, [target], SCDAOptions(enabled=True, diagnostics=True))[0]
        self.assertEqual(legacy.audit["per_gt"][0]["reg_transition_count"], 0)

    def test_missing_medium_never_refills_from_core(self):
        for medium_count in (0, 1):
            with self.subTest(medium_count=medium_count):
                values = [1.] * 16
                if medium_count: values[15] = .5
                _, _, assigned = self.case(values)
                entry = assigned.audit["per_gt"][0]
                self.assertEqual(len(assigned.reg.token_indices), 2 + medium_count)
                self.assertEqual(entry["reg_transition_shortfall"], 2 - medium_count)
                self.assertEqual(entry["reg_core_shortfall"], 0)

    def test_missing_core_never_refills_from_medium(self):
        for core_count in (0, 1):
            values = [.5] * 16
            if core_count: values[5] = 1.
            _, _, assigned = self.case(values)
            self.assertEqual(len(assigned.cls.token_indices), core_count)
            self.assertEqual(len(assigned.reg.token_indices), core_count + 2)
            self.assertEqual(assigned.audit["per_gt"][0]["reg_core_shortfall"], 2-core_count)

    def test_medium_ranking_distance_then_lexicographic_tie(self):
        _, _, assigned = self.case([.5] * 16)
        # Central four cells have equal coverage and distance. u before v.
        self.assertEqual(assigned.reg.token_indices.tolist(), [5, 9])
        _, _, assigned = self.case([0.] * 16)
        self.assertEqual(len(assigned.reg.token_indices), 0)

    def test_medium_receives_only_box_auxiliary_gradients(self):
        payload, target, assigned = self.case([1.] * 12 + [.75, .5, .25, 0.])
        losses = scda_losses(payload, [target], [assigned], self.options, 5, 2)
        sum(losses.losses.values()).backward()
        self.assertEqual(float(payload["raw_cls"].grad[0, [13, 14]].abs().sum()), 0)
        self.assertGreater(float(payload["raw_box_sigmoid"].grad[0, [13, 14]].abs().sum()), 0)
        self.assertEqual(float(payload["raw_box_sigmoid"].grad[0, 0].abs().sum()), 0)

    def test_shortfall_box_loss_averages_actual_three_points(self):
        from methods.scda_proposal.scda_loss import aligned_giou, cxcywh_to_xyxy
        payload, target, assigned = self.case([1.] * 15 + [.5])
        p = assigned.reg.token_indices
        self.assertEqual(len(p), 3)
        boxes = payload["raw_box_sigmoid"][0, p] * assigned.box_scale[p]
        truth = target.boxes.repeat(3, 1)
        expected = (5 * (boxes - truth).abs().sum(-1) +
                    2 * (1 - aligned_giou(cxcywh_to_xyxy(boxes), cxcywh_to_xyxy(truth)))).mean()
        actual = scda_losses(payload, [target], [assigned], self.options, 5, 2)
        torch.testing.assert_close(actual.unweighted["box"], expected)

    def test_v2_validity_owner_and_source_reliability(self):
        for case in ("empty", "unreliable", "invalid", "padding", "overlap"):
            payload, target = fixture(gt_count=0 if case == "empty" else (2 if case == "overlap" else 1))
            if case == "unreliable": target.reliable[:] = False
            if case == "invalid": payload["valid_token_mask"][:] = False
            if case == "padding": target.valid_pixels[:] = False
            assigned = assign_scda(payload, [target], self.options)[0]
            if case == "overlap":
                self.assertTrue((assigned.reg.gt_indices == 0).all())
            else:
                self.assertEqual(len(assigned.reg.token_indices), 0, case)

    def test_unknown_or_unconfirmed_policy_parameters_rejected(self):
        for changes in ({"reg_selection": "other"}, {"k_reg": 5}, {"k_cls": 1}, {"tau_reg": .1}):
            with self.assertRaises(ValueError):
                dataclasses.replace(self.options, **changes).validate()


class QualityAssignmentTests(unittest.TestCase):
    """quality_iou keeps the confirmed pool and quotas; only the ranking input
    changes from mask coverage to predicted-box IoU (detached, FP32)."""

    def case(self, coverage_values, boxes_by_token=None, topk=(0,), options=None):
        payload, target = fixture()
        target.boxes[:] = torch.tensor([.5, .5, 1., 1.])
        target.masks[:] = False
        for token, value in enumerate(coverage_values):
            y, x = divmod(token, 4)
            for pixel in range(round(value * 4)):
                dy, dx = divmod(pixel, 2)
                target.masks[0, y * 2 + dy, x * 2 + dx] = True
        if boxes_by_token is not None:
            payload["raw_box_sigmoid"] = payload["raw_box_sigmoid"].detach().clone()
            for token, box in boxes_by_token.items():
                payload["raw_box_sigmoid"][0, token] = torch.tensor(box)
        payload["topk_indices"] = torch.tensor([list(topk)])
        opts = options or SCDAOptions(enabled=True, diagnostics=True, reg_selection="quality_iou")
        return payload, target, assign_scda(payload, [target], opts)[0]

    def test_quality_ranking_changes_set_and_order_vs_coverage(self):
        # Top row fully covered but their default anchor boxes barely overlap the
        # GT; token 5 has 0.5 coverage yet its predicted box equals the GT.
        # topk=(15,) keeps the far corner out of the pool for this fixture.
        coverage = [1., 1., 1., 1.] + [0.] * 12
        coverage[5] = .5
        perfect = [.5, .5, 1., 1.]
        _, _, quality = self.case(coverage, boxes_by_token={5: perfect}, topk=(15,))
        _, _, coverage_rule = self.case(coverage, topk=(15,),
                                        options=SCDAOptions(enabled=True, diagnostics=True))
        self.assertEqual(quality.reg.token_indices.tolist(), [5, 1, 2, 0])
        self.assertEqual(coverage_rule.reg.token_indices.tolist(), [1, 2, 0, 3])
        self.assertAlmostEqual(quality.audit["per_gt"][0]["reg_iou"][0], 1.0, places=5)
        self.assertEqual(quality.cls.token_indices.tolist(), [1, 2])

    def test_equal_iou_breaks_by_distance_then_lexicographic(self):
        perfect = [.5, .5, 1., 1.]
        coverage = [0.] * 16
        coverage[1] = coverage[5] = .5
        _, _, assigned = self.case(coverage, boxes_by_token={5: perfect, 1: perfect}, topk=(15,))
        # Equal IoU: nearer center (token 5, d=.031) before token 1 (d=.156).
        self.assertEqual(assigned.reg.token_indices[:2].tolist(), [5, 1])
        coverage[1] = 0.
        coverage[6] = .5
        _, _, assigned = self.case(coverage, boxes_by_token={5: perfect, 6: perfect}, topk=(15,))
        # Equal IoU and distance: (level,u=x,v=y) puts token 5 before 6.
        self.assertEqual(assigned.reg.token_indices[:2].tolist(), [5, 6])

    def test_pool_threshold_and_topq_exclusion_still_apply(self):
        perfect = [.5, .5, 1., 1.]
        # Perfect box but zero mask coverage stays outside the confirmed pool.
        _, _, assigned = self.case([0.] * 16, boxes_by_token={5: perfect})
        self.assertEqual(len(assigned.reg.token_indices), 0)
        # A Top-Q token is never selected, regardless of its box quality.
        _, _, assigned = self.case([1.] * 16, boxes_by_token={5: perfect}, topk=(5,))
        self.assertNotIn(5, assigned.reg.token_indices.tolist())

    def test_boxonly_drops_coverage_gate_but_keeps_rest(self):
        perfect = [.5, .5, 1., 1.]
        # Zero-coverage tokens enter the pool and win purely by IoU ranking;
        # the high-coverage far corner loses despite its coverage.
        coverage = [0.] * 16
        coverage[15] = 1.
        boxes = {5: perfect, 6: perfect, 7: perfect}
        boxonly_opts = SCDAOptions(enabled=True, diagnostics=True, reg_selection="quality_iou_boxonly")
        _, _, boxonly = self.case(coverage, boxes_by_token=boxes, topk=(15,), options=boxonly_opts)
        self.assertEqual(boxonly.reg.token_indices[:3].tolist(), [5, 6, 7])
        self.assertNotIn(15, boxonly.reg.token_indices.tolist())
        # The coverage gate keeps the same tokens out under quality_iou.
        gated_opts = SCDAOptions(enabled=True, diagnostics=True, reg_selection="quality_iou")
        _, _, gated = self.case(coverage, boxes_by_token=boxes, topk=(15,), options=gated_opts)
        self.assertNotIn(5, gated.reg.token_indices.tolist())
        self.assertNotIn(6, gated.reg.token_indices.tolist())
        # Wider owner set does not change the cls core under this single-GT target.
        self.assertEqual(boxonly.cls.token_indices.tolist(), gated.cls.token_indices.tolist())
        self.assertAlmostEqual(boxonly.audit["per_gt"][0]["reg_iou"][0], 1.0, places=5)

    def test_defaults_and_validation(self):
        defaults = SCDAOptions()
        self.assertEqual(defaults.reg_selection, "coverage_topk")
        self.assertEqual(defaults.cls_loss_form, "bce")
        SCDAOptions(reg_selection="quality_iou").validate()
        SCDAOptions(reg_selection="quality_iou_boxonly").validate()
        for changes in ({"reg_selection": "quality_iou", "k_reg": 5},
                        {"reg_selection": "quality_iou_boxonly", "k_reg": 5},
                        {"cls_loss_form": "mse"}, {"focal_gamma": -1.}, {"focal_alpha": float("inf")}):
            with self.assertRaises(ValueError):
                SCDAOptions(**changes).validate()


class FocalClsLossTests(unittest.TestCase):
    """focal_positive = native-aligned positive focal term; opt-in, FP32."""

    def test_value_matches_native_positive_focal_term(self):
        from methods.scda_proposal.scda_loss import positive_cls_loss
        options = SCDAOptions(cls_loss_form="focal_positive")
        logits = torch.tensor([-7., -0.5, 0., 1.5, 6.])
        p = torch.sigmoid(logits)
        expected = options.focal_alpha * (1 - p) ** options.focal_gamma * torch.nn.functional.softplus(-logits)
        torch.testing.assert_close(positive_cls_loss(logits, options), expected)

    def test_downweights_confident_tokens_keeps_hard_ones(self):
        from methods.scda_proposal.scda_loss import positive_cls_loss
        focal, bce = SCDAOptions(cls_loss_form="focal_positive"), SCDAOptions()
        hard = positive_cls_loss(torch.tensor([-8.]), focal) / positive_cls_loss(torch.tensor([-8.]), bce)
        easy = positive_cls_loss(torch.tensor([8.]), focal) / positive_cls_loss(torch.tensor([8.]), bce)
        self.assertAlmostEqual(float(hard), 0.25, places=3)
        self.assertLess(float(easy), 1e-3)

    def test_per_gt_mean_and_zero_lambda_exact_zero(self):
        from methods.scda_proposal.scda_loss import positive_cls_loss, scda_losses as _  # noqa: F401
        payload, target = fixture()
        assigned = assign_scda(payload, [target], SCDAOptions(enabled=True))[0]
        options = SCDAOptions(enabled=True, cls_loss_form="focal_positive")
        result = scda_losses(payload, [target], [assigned], options, 5, 2)
        p = assigned.cls.token_indices
        torch.testing.assert_close(result.unweighted["cls"],
                                   positive_cls_loss(payload["raw_cls"][0, p, 0].float(), options).mean())
        zero = dataclasses.replace(options, lambda_cls=0.0)
        zero_result = scda_losses(payload, [target], [assigned], zero, 5, 2)
        self.assertEqual(float(zero_result.losses["loss_scda_cls"]), 0.0)
        payload["raw_cls"].grad = None
        sum(zero_result.losses.values()).backward()
        self.assertEqual(float(payload["raw_cls"].grad.abs().sum()), 0)


class LossTests(unittest.TestCase):
    def test_positive_class_only_and_box_gradients(self):
        payload, target = fixture()
        assigned = assign_scda(payload, [target], SCDAOptions(enabled=True))[0]
        result = scda_losses(payload, [target], [assigned], SCDAOptions(), 5, 2)
        sum(result.losses.values()).backward()
        self.assertGreater(payload["raw_cls"].grad[..., 0].abs().sum(), 0)
        self.assertEqual(float(payload["raw_cls"].grad[..., 1].abs().sum()), 0)
        self.assertGreater(payload["raw_box_sigmoid"].grad.abs().sum(), 0)

    def test_no_positives_zero_is_finite_and_attached(self):
        payload, target = fixture(gt_count=0)
        payload["raw_cls"].data[0, 0] = float("inf")
        assigned = assign_scda(payload, [target], SCDAOptions(enabled=True))[0]
        result = scda_losses(payload, [target], [assigned], SCDAOptions(), 5, 2)
        loss = sum(result.losses.values())
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss), 0)
        loss.backward()
        self.assertEqual(float(payload["raw_cls"].grad.abs().sum()), 0)

    def test_per_gt_mean_not_per_token_mean(self):
        logits = torch.tensor([[[0.], [2.], [4.]]], requires_grad=True)
        boxes = torch.tensor([[[.5, .5, .4, .4]]]*3).transpose(0, 1).requires_grad_()
        payload = {"raw_cls": logits, "raw_box_sigmoid": boxes}
        _, target = fixture(gt_count=2)
        cls = TaskAssignment(torch.tensor([0, 1, 2]), torch.tensor([0, 0, 1]))
        empty = TaskAssignment(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
        assigned = ImageAssignment(cls, empty, torch.ones(3, 4))
        observed = scda_losses(payload, [target], [assigned], SCDAOptions(), 5, 2).unweighted["cls"]
        expected = ((torch.nn.functional.softplus(-logits[0, :2, 0]).mean()) + torch.nn.functional.softplus(-logits[0, 2, 0])) / 2
        torch.testing.assert_close(observed, expected)

    def test_gloo_two_ranks_one_and_both_empty_match_serial_gradient(self):
        with tempfile.TemporaryDirectory() as directory:
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            mp.spawn(gloo_rank, args=(port, directory), nprocs=2, join=True)
            serial = torch.nn.Linear(1, 5, bias=False)
            with torch.no_grad(): serial.weight.copy_(torch.tensor([[.2], [.3], [.4], [.5], [.6]]))
            cases = [rank_case(serial, rank) for rank in (0, 1)]
            payload = {key: torch.cat([case[0][key] for case in cases]) for key in cases[0][0]}
            result = scda_losses(payload, [case[1] for case in cases], [case[2] for case in cases], SCDAOptions(), 5, 2)
            sum(result.losses.values()).backward()
            for rank in (0, 1):
                measured = torch.load(Path(directory) / f"rank{rank}.pt", weights_only=True)
                torch.testing.assert_close(measured["False"]["grad"], serial.weight.grad)
                torch.testing.assert_close(measured["False"]["counts"], torch.tensor([2., 2.]))
                self.assertEqual(float(measured["True"]["grad"].abs().sum()), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
