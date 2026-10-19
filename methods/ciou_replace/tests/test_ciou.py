import copy
import unittest
import torch
from detectron2.layers.losses import ciou_loss
from maskdino.modeling.criterion import SetCriterion
from maskdino.modeling.matcher import HungarianMatcher
from methods.ciou_replace.loss import ciou_pairs, reference_record
from methods.ciou_replace.model import CIoUCriterion


class CIoUTests(unittest.TestCase):
    def test_pinned_reference(self):
        self.assertFalse(reference_record()["iou_gate"])

    def test_paired_values_and_gradients_match_reference(self):
        a = [[0., 0., 1., 1.], [0., 0., .3, .7], [-.2, .1, .8, .4],
             [.1, .1, .10001, .10002], [.2, .3, .8, .9]]
        b = [[0., 0., 1., 1.], [.6, .5, .9, .9], [.1, .2, .5, .7],
             [.10001, .1, .10003, .10005], [.3, .1, .7, .6]]
        for dtype in (torch.float32, torch.float64):
            pred = torch.tensor(a, dtype=dtype, requires_grad=True)
            target = torch.tensor(b, dtype=dtype)
            value = ciou_pairs(pred, target)
            expected = ciou_loss(pred, target, reduction="none", eps=1e-7)
            self.assertEqual(value.shape, (5,))
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
            torch.testing.assert_close(torch.autograd.grad(value.sum(), pred, retain_graph=True)[0],
                                       torch.autograd.grad(expected.sum(), pred)[0], rtol=0, atol=0)

    def test_empty_is_connected_and_invalid_not_silently_fixed(self):
        empty = torch.empty(0, 4, requires_grad=True)
        zero = ciou_pairs(empty, empty).sum()
        self.assertEqual(float(zero), 0.)
        self.assertEqual(torch.autograd.grad(zero, empty)[0].shape, (0, 4))
        target = torch.tensor([[0., 0., 1., 1.]])
        for bad in ([[0., 0., 0., 1.]], [[1., 0., 0., 1.]], [[0., 0., float('nan'), 1.]]):
            pred = torch.tensor(bad, requires_grad=True)
            original = pred.detach().clone()
            with self.assertRaises(FloatingPointError):
                ciou_pairs(pred, target)
            torch.testing.assert_close(pred.detach(), original, equal_nan=True)

    def test_frozen_alpha_finite_difference(self):
        from math import pi
        x = torch.tensor([[.13, .21, .67, .86]], dtype=torch.float64, requires_grad=True)
        gt = torch.tensor([[.22, .15, .82, .71]], dtype=torch.float64)
        def formula(z, fixed_alpha=None):
            wh = z[:, 2:] - z[:, :2]; tg = gt[:, 2:] - gt[:, :2]
            inter = (torch.minimum(z[:, 2:], gt[:, 2:]) - torch.maximum(z[:, :2], gt[:, :2])).clamp_min(0).prod(-1)
            iou = inter / (wh.prod(-1) + tg.prod(-1) - inter + 1e-7)
            diag = (torch.maximum(z[:, 2:], gt[:, 2:]) - torch.minimum(z[:, :2], gt[:, :2])).square().sum(-1) + 1e-7
            rho = ((z[:, :2] + z[:, 2:] - gt[:, :2] - gt[:, 2:]) / 2).square().sum(-1)
            v = 4 / pi**2 * (torch.atan(tg[:, 0]/tg[:, 1]) - torch.atan(wh[:, 0]/wh[:, 1])).square()
            alpha = (v / (1-iou+v+1e-7)).detach() if fixed_alpha is None else fixed_alpha
            return 1-iou+rho/diag+alpha*v, alpha
        expected, alpha = formula(x)
        actual = ciou_pairs(x, gt)
        torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
        grad = torch.autograd.grad(actual.sum(), x)[0]
        eps = 1e-6
        for j in range(4):
            delta = torch.zeros_like(x); delta[0, j] = eps
            numeric = (formula(x.detach()+delta, alpha)[0]-formula(x.detach()-delta, alpha)[0])/(2*eps)
            self.assertAlmostEqual(float(grad[0, j]), float(numeric), places=7)

    def test_native_other_losses_matching_rng_and_multi_gt(self):
        kwargs = dict(num_classes=1, matcher=HungarianMatcher(cost_class=4, cost_mask=5,
            cost_dice=5, cost_box=5, cost_giou=2, num_points=32), weight_dict={}, eos_coef=.1,
            losses=["labels", "masks", "boxes"], num_points=32, oversample_ratio=3.,
            importance_sample_ratio=.75, dn="no", dn_losses=[], panoptic_on=False, semantic_ce_loss=False)
        native = SetCriterion(**kwargs)
        method = CIoUCriterion(**kwargs)
        torch.manual_seed(42)
        def output():
            return {"pred_logits": torch.randn(1, 6, 1, requires_grad=True),
                    "pred_masks": torch.randn(1, 6, 8, 8, requires_grad=True),
                    "pred_boxes": torch.rand(1, 6, 4, requires_grad=True)*.5+.25}
        outputs = output(); outputs["aux_outputs"] = [output(), output()]; outputs["interm_outputs"] = output()
        targets = [{"labels": torch.tensor([0, 0]), "masks": torch.ones(2, 32, 32),
                    "boxes": torch.tensor([[.5, .5, .5, .5], [.3, .4, .2, .3]])}]
        state = torch.get_rng_state(); expected = native(outputs, targets); after = torch.get_rng_state()
        observed = []
        def inspect(criterion, suffix, out, ts, indices, normalizer, values):
            observed.append(suffix)
            self.assertEqual(sum(len(j) for _, j in indices), 2)
            self.assertGreater(float(torch.autograd.grad(values["loss_giou"], out["pred_boxes"], retain_graph=True)[0].abs().sum()), 0)
        method.audit_callback = inspect
        torch.set_rng_state(state); actual = method(outputs, targets)
        torch.testing.assert_close(after, torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(set(actual), set(expected))
        self.assertEqual(set(observed), {"", "_0", "_1", "_interm"})
        for key in expected:
            if not key.startswith("loss_giou"):
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        self.assertEqual(set(native.state_dict()), set(method.state_dict()))

    def test_native_detached_initialization_is_reported_not_fabricated(self):
        from types import SimpleNamespace
        from methods.ciou_replace.audit import BranchAudit
        model = torch.nn.Module(); model.bbox_embed = torch.nn.Linear(2, 4)
        learned = model.bbox_embed(torch.ones(1, 2)).sigmoid()
        stacked = torch.stack([torch.full_like(learned, .4), learned])
        output = {"pred_boxes": stacked[:1]}
        target = [{"boxes": torch.tensor([[.5, .5, .2, .3]])}]
        criterion = SimpleNamespace(weight_dict={"loss_giou_0": 2., "loss_bbox_0": 5.},
            _get_src_permutation_idx=lambda idx: (torch.tensor([0]), torch.tensor([0])))
        from maskdino.utils import box_ops
        a = box_ops.box_cxcywh_to_xyxy(output["pred_boxes"][0])
        b = box_ops.box_cxcywh_to_xyxy(target[0]["boxes"])
        raw = {"loss_giou": ciou_pairs(a, b).sum(), "loss_bbox": (output["pred_boxes"][0]-target[0]["boxes"]).abs().sum()}
        audit = BranchAudit(model)
        audit(criterion, "_0", output, target, [(torch.tensor([0]), torch.tensor([0]))], 1., raw)
        record = audit.verify({"loss_giou_0": raw["loss_giou"]*2})
        self.assertIn("BOTH CIoU/GIoU", record["keys_weights_gradients"][0]["head_gradient_status"])


if __name__ == "__main__":
    unittest.main()
