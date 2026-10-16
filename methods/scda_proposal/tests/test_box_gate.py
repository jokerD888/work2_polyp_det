"""Box-Gated geometry, selection, fitting and gradient checks; CPU only."""
import unittest

import torch

from methods.scda_proposal.box_gate import (candidate_geometry, shrink_box_contains,
                                           calibration_observation)
from methods.scda_proposal.box_gate_calibration import RHO_GRID, fit_count_calibration
from methods.scda_proposal.config import SCDAOptions
from methods.scda_proposal.scda_assigner import assign_scda
from methods.scda_proposal.scda_loss import scda_losses
from methods.scda_proposal.tests.test_cpu import fixture


class BoxGateTests(unittest.TestCase):
    def options(self, **changes):
        values = dict(enabled=True, diagnostics=True, reg_selection="quality_iou_boxgate",
                      box_gate_rho=.5, lambda_cls=0., lambda_box=.05)
        return SCDAOptions(**dict(values, **changes)).validate()

    def test_half_open_anisotropic_shrink_geometry(self):
        box = torch.tensor([.5, .5, .8, .4], dtype=torch.float64)
        points = torch.tensor([[.3, .4], [.7, .5], [.5, .6], [.5, .5], [.29, .5]],
                              dtype=torch.float64)
        self.assertEqual(shrink_box_contains(points, box, .5).tolist(),
                         [True, False, False, True, False])

    def test_mask_shape_does_not_change_box_gate_selection_or_gradients(self):
        payload, target = fixture()
        target.boxes[:] = torch.tensor([.5, .5, 1., 1.])
        first = assign_scda(payload, [target], self.options())[0]
        first_loss = scda_losses(payload, [target], [first], self.options(), 5, 2)
        first_grad = torch.autograd.grad(sum(first_loss.losses.values()),
                                         (payload["raw_cls"], payload["raw_box_sigmoid"]))
        # Preserve common nonempty-annotation validity, radically change shape.
        target.masks[:] = False
        target.masks[0, 0, 0] = True
        second = assign_scda(payload, [target], self.options())[0]
        second_loss = scda_losses(payload, [target], [second], self.options(), 5, 2)
        second_grad = torch.autograd.grad(sum(second_loss.losses.values()),
                                          (payload["raw_cls"], payload["raw_box_sigmoid"]))
        self.assertEqual(first.reg.token_indices.tolist(), second.reg.token_indices.tolist())
        self.assertEqual(set(first.reg.token_indices.tolist()), {5, 6, 9, 10})
        self.assertEqual(len(second.cls.token_indices), 0)
        for a, b in zip(first_grad, second_grad):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(float(second_grad[0].abs().sum()), 0.)
        self.assertGreater(float(second_grad[1].abs().sum()), 0.)

    def test_shrink_gate_is_not_the_existing_full_boxonly_pool(self):
        payload, target = fixture()
        target.boxes[:] = torch.tensor([.5, .5, 1., 1.])
        # Off-center perfect proposal wins Q-BoxOnly but is spatially disallowed
        # by the new shrunken gate, despite maximum predicted IoU.
        with torch.no_grad():
            payload["raw_box_sigmoid"][0, 1] = target.boxes[0]
        box = assign_scda(payload, [target], self.options())[0]
        full = assign_scda(payload, [target], SCDAOptions(enabled=True,
                           reg_selection="quality_iou_boxonly", lambda_cls=0.))[0]
        self.assertNotIn(1, box.reg.token_indices.tolist())
        self.assertEqual(full.reg.token_indices[0].item(), 1)

    def test_iou_ranking_ties_and_topq_exclusion(self):
        payload, target = fixture()
        target.boxes[:] = torch.tensor([.5, .5, 1., 1.])
        with torch.no_grad():
            payload["raw_box_sigmoid"][0, 10] = target.boxes[0]
        payload["topk_indices"] = torch.tensor([[6]])
        chosen = assign_scda(payload, [target], self.options())[0]
        self.assertEqual(chosen.reg.token_indices.tolist(), [10, 5, 9])
        self.assertEqual(chosen.audit["per_gt"][0]["reg_shortfall"], 1)
        self.assertEqual(chosen.audit["per_gt"][0]["reg_iou"][0], 1.)

    def test_full_cell_padding_empty_gt_and_reliability_are_preserved(self):
        for case in ("padding", "empty", "unreliable", "empty_mask"):
            payload, target = fixture(gt_count=0 if case == "empty" else 1)
            if case == "padding": target.valid_pixels[:] = False
            if case == "unreliable": target.reliable[:] = False
            if case == "empty_mask": target.masks[:] = False
            assigned = assign_scda(payload, [target], self.options())[0]
            self.assertEqual(len(assigned.reg.token_indices), 0, case)
            loss = sum(scda_losses(payload, [target], [assigned], self.options(), 5, 2).losses.values())
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertEqual(float(payload["raw_box_sigmoid"].grad.abs().sum()), 0.)

    def test_original_multi_gt_is_skipped_and_unapproved_reliable_multi_rejected(self):
        payload, target = fixture(gt_count=2, reliable=False)
        self.assertEqual(len(assign_scda(payload, [target], self.options())[0].reg.token_indices), 0)
        target.reliable[:] = True
        with self.assertRaises(ValueError):
            assign_scda(payload, [target], self.options())

    def test_calibration_counts_match_runtime_gate_and_exclude_topq(self):
        payload, target = fixture()
        target.boxes[:] = torch.tensor([.5, .5, 1., 1.])
        payload["topk_indices"] = torch.tensor([[5]])
        observation = calibration_observation(payload, 0, target)
        self.assertEqual(observation["full_count"], 15)
        self.assertEqual(observation["mask_count"], 15)
        self.assertEqual(observation["box_counts"][49], 3)
        self.assertEqual(observation["box_counts"][-1], 15)
        self.assertEqual(observation["box_counts_per_level_by_rho"][49], [3])
        assigned = assign_scda(payload, [target], self.options())[0]
        self.assertEqual(assigned.audit["per_gt"][0]["n_reg_eligible"], 3)
        _, _, _, center, _, available = candidate_geometry(payload, 0, target)
        for i, rho in enumerate(RHO_GRID):
            expected = int((available & shrink_box_contains(center, target.boxes[0], rho)).sum())
            self.assertEqual(observation["box_counts"][i], expected)

    def test_invalid_or_unconfirmed_options_fail(self):
        for changes in ({"box_gate_rho": None}, {"box_gate_rho": 0.}, {"box_gate_rho": 1.1},
                        {"box_gate_rho": float("nan")}, {"box_gate_rho": True},
                        {"lambda_cls": .05}, {"lambda_box": .1}, {"mask_policy": "explicit"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.options(**changes)
        with self.assertRaises(ValueError):
            SCDAOptions(box_gate_rho=.5).validate()

    def test_real_cpu_boxgate_model_has_no_new_parameters_or_rng_drift(self):
        import gc
        from pathlib import Path
        import sys
        import tempfile
        root = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(root / "third_party/MaskDINO"))
        import train_net
        from detectron2.modeling import build_model
        from methods.scda_proposal.experiment import load_experiment
        from methods.scda_proposal.integration import register
        from methods.scda_proposal.precheck import build_cfg
        register(root / "third_party/MaskDINO")
        experiment = load_experiment(root / "methods/scda_proposal/configs/kvasir_freg_quality.yaml")
        experiment["scda"].update(reg_selection="quality_iou_boxgate", box_gate_rho=.5)
        with tempfile.TemporaryDirectory() as output:
            native_cfg = build_cfg(experiment, False, output)
            torch.manual_seed(42)
            native = build_model(native_cfg)
            reference = {name: tensor.clone() for name, tensor in native.state_dict().items()}
            del native
            cfg = build_cfg(experiment, True, output)
            torch.manual_seed(42)
            modified = build_model(cfg)
            self.assertEqual(reference.keys(), modified.state_dict().keys())
            self.assertTrue(all(torch.equal(value, modified.state_dict()[name])
                                for name, value in reference.items()))
            optimizer = train_net.Trainer.build_optimizer(cfg, modified)
            actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
            self.assertEqual(len(actual), len(set(actual)))
            self.assertEqual(set(actual), {id(p) for p in modified.parameters() if p.requires_grad})
            self.assertEqual(modified.scda_options.box_gate_rho, .5)
            self.assertEqual(modified.scda_options.lambda_cls, 0.)
            del reference, optimizer, modified
            gc.collect()
        self.assertFalse(torch.cuda.is_initialized())


class CountCalibrationTests(unittest.TestCase):
    def test_pooled_count_match_not_mean_of_image_retention_ratios(self):
        # Small image retention is 100%, large image retention 10%; pooled is
        # 19/109, NOT 55%. Four capped positives would hide this difference.
        rows = [{"full_count": 9, "mask_count": 9,
                 "box_counts": [0] * 49 + [9] * 51},
                {"full_count": 100, "mask_count": 10,
                 "box_counts": [0] * 49 + [10] * 51}]
        fit = fit_count_calibration(rows)
        self.assertEqual(fit["rho"], .5)
        self.assertEqual(fit["mask_count"], fit["box_count"])
        self.assertAlmostEqual(fit["mask_retention"], 19 / 109)

    def test_tie_chooses_smaller_rho_and_residual_is_reported(self):
        fit = fit_count_calibration([{"full_count": 20, "mask_count": 5,
                                      "box_counts": [4] * 50 + [6] * 50}])
        self.assertEqual(fit["rho"], .01)
        self.assertEqual(fit["count_difference"], -1)
        self.assertEqual(fit["relative_count_error"], -.2)

    def test_no_supervision_or_invalid_counts_cannot_generate_rho(self):
        for rows in ([], [{"full_count": 10, "mask_count": 0, "box_counts": [0] * 100}],
                     [{"full_count": 10, "mask_count": 5, "box_counts": [0] * 100}],
                     [{"full_count": 10, "mask_count": 5, "box_counts": [6, 5] + [10] * 98}]):
            with self.assertRaises(ValueError):
                fit_count_calibration(rows)


if __name__ == "__main__":
    unittest.main()
