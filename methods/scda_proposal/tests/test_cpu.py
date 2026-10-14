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
