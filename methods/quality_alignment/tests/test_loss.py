import ast
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F
from torchvision.ops import box_iou, box_convert

from methods.quality_alignment.loss import quality_loss, aligned_iou, REFERENCE_SHA256
from polypbench.io import sha256


def reference():
    path = Path(__file__).resolve().parents[1] / "reference/align_losses.py"
    if sha256(path) != REFERENCE_SHA256:
        raise RuntimeError("Pinned reference drift")
    tree = ast.parse(path.read_text())
    selected = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("get_local_rank", "IA_BCE_loss")], type_ignores=[])
    namespace = {"torch": torch, "box_iou": lambda a,b: (box_iou(a,b), None),
                 "box_cxcywh_to_xyxy": lambda x: box_convert(x, in_fmt="cxcywh", out_fmt="xyxy")}
    exec(compile(selected, str(path), "exec"), namespace)
    return namespace["IA_BCE_loss"]


class QualityLossTests(unittest.TestCase):
    def test_reference_values_gradients_unsorted_multi_gt_and_classes(self):
        fn = reference()
        generator = torch.Generator().manual_seed(80)
        for nclass in (1, 3):
            for repetition in range(5):
                z = torch.randn((2, 4, nclass), generator=generator, dtype=torch.float64, requires_grad=True)
                boxes = torch.rand((3, 4), generator=generator, dtype=torch.float64)
                boxes[:, 2:] = boxes[:, 2:] * .4 + .1
                target = boxes.detach().clone(); target[:, :2] += .02
                b, q = torch.tensor([0, 0, 1]), torch.tensor([2, 0, 3])
                c = torch.tensor([0, nclass - 1, 0])
                indices = [(torch.tensor([2, 0]), torch.tensor([1, 0])), (torch.tensor([3]), torch.tensor([0]))]
                expected, _ = fn(z, (b, q, c), boxes, target, indices, 3., .25, 2., 1)
                actual = quality_loss(z, boxes, target, b, q, c, 3.)["loss"]
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                ga = torch.autograd.grad(actual, z, retain_graph=True)[0]
                ge = torch.autograd.grad(expected, z)[0]
                torch.testing.assert_close(ga, ge, rtol=1e-11, atol=1e-12)

    def test_targets_detached_and_fixed_target_finite_difference(self):
        z = torch.tensor([[[-.4], [1.2]]], dtype=torch.float64, requires_grad=True)
        boxes = torch.tensor([[.4, .4, .2, .2]], dtype=torch.float64, requires_grad=True)
        truth = torch.tensor([[.42, .4, .2, .2]], dtype=torch.float64)
        b = q = c = torch.tensor([0])
        parts = quality_loss(z, boxes, truth, b, q, c, 1.)
        self.assertFalse(parts['target'].requires_grad)
        self.assertIsNone(torch.autograd.grad(parts['loss'], boxes, allow_unused=True, retain_graph=True)[0])
        actual = torch.autograd.grad(parts['loss'], z)[0]
        target = parts['target'][0].item()
        def scalar(x):
            return F.binary_cross_entropy_with_logits(x[0, 0, 0], x.new_tensor(target)) + x[0, 1, 0].sigmoid().square() * F.softplus(x[0, 1, 0])
        for i in range(2):
            plus, minus = z.detach().clone(), z.detach().clone()
            plus[0, i, 0] += 1e-5; minus[0, i, 0] -= 1e-5
            self.assertAlmostEqual(actual[0, i, 0].item(), ((scalar(plus) - scalar(minus)) / 2e-5).item(), places=8)

    def test_extreme_logits_zero_iou_and_empty_matching_are_finite(self):
        z = torch.tensor([[[-1000.], [1000.], [-.5]]], requires_grad=True)
        box = torch.tensor([[.1, .1, .1, .1]])
        truth = torch.tensor([[.9, .9, .1, .1]])
        idx = torch.tensor([0])
        p = quality_loss(z, box, truth, idx, idx, idx, 1.)
        self.assertAlmostEqual(p['target'].item(), .01, places=7)
        self.assertTrue(torch.isfinite(p['loss']))
        self.assertTrue(torch.isfinite(torch.autograd.grad(p['loss'], z)[0]).all())
        empty = torch.empty(0, dtype=torch.long)
        p = quality_loss(z, box[:0], truth[:0], empty, empty, empty, 1.)
        self.assertEqual(p['stats']['positive_count'], 0)
        self.assertEqual(p['positive'].item(), 0)
        self.assertGreater(p['negative'].item(), 0)

    def test_iou_endpoints_and_degenerate_prediction(self):
        box = torch.tensor([[.5, .5, .2, .2], [.5, .5, 0., 0.]])
        truth = torch.tensor([[.5, .5, .2, .2], [.5, .5, .2, .2]])
        torch.testing.assert_close(aligned_iou(box, truth), torch.tensor([1., 0.]), rtol=1e-5, atol=1e-6)
        with self.assertRaises(ValueError):
            aligned_iou(box, box)

    def test_negative_weight_is_differentiated(self):
        z = torch.tensor([[[.3]]], dtype=torch.float64, requires_grad=True)
        empty = torch.empty(0, dtype=torch.long)
        out = quality_loss(z, torch.empty(0, 4), torch.empty(0, 4), empty, empty, empty, 1.)
        expected = z.sigmoid().pow(3) + 2 * z.sigmoid().square() * (1 - z.sigmoid()) * F.softplus(z)
        torch.testing.assert_close(torch.autograd.grad(out['loss'], z)[0], expected)


if __name__ == '__main__':
    unittest.main()
