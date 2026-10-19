import importlib.util
from dataclasses import replace
from pathlib import Path
import unittest
import tempfile
from types import SimpleNamespace

import torch

from methods.mssl.config import MSSLOptions
from methods.mssl.loss import side_losses
from methods.mssl.gradient_probe import GradientProbe


class AttributionTests(unittest.TestCase):
    def setUp(self):
        self.mask = torch.zeros(32, 32)
        self.mask[6:25, 8:24] = 1
        self.mask[6:12, 8:18] = 0
        self.target = torch.tensor([6., 4., 27., 28.])

    def test_static_is_local_second_order_with_normalized_coordinates(self):
        options = MSSLOptions()
        size = (self.target[2:] - self.target[:2]).repeat(2)
        for sign in (-1, 1):
            pred = (self.target + sign * .0005 * size).requires_grad_()
            cdf = side_losses(pred, self.target, self.mask, options)
            static = side_losses(pred, self.target, self.mask, replace(options, loss_form="static"))
            torch.testing.assert_close(cdf, static, rtol=.02, atol=1e-10)
            g1 = torch.autograd.grad(cdf.sum(), pred, retain_graph=True)[0]
            g2 = torch.autograd.grad(static.sum(), pred)[0]
            torch.testing.assert_close(g1, g2, rtol=.03, atol=1e-8)

    def test_static_zero_target_and_finite_gradient_at_large_error(self):
        opt = replace(MSSLOptions(), loss_form="static")
        pred = self.target.clone().requires_grad_()
        zero = side_losses(pred, self.target, self.mask, opt).sum()
        self.assertEqual(float(zero), 0.)
        torch.testing.assert_close(torch.autograd.grad(zero, pred)[0], torch.zeros(4))
        pred = (self.target + 50).requires_grad_()
        loss = side_losses(pred, self.target, self.mask, opt).sum()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(torch.autograd.grad(loss, pred)[0]).all())

    def test_cdf_matches_actual_frozen_v1_bit_for_bit(self):
        path = Path("runs/mssl-implementation-20261018/campaign-r2/source/methods/mssl/loss.py")
        if not path.exists():
            self.skipTest("Historical training bundle is a local artifact")
        spec = importlib.util.spec_from_file_location("frozen_mssl_loss", path)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        for support in ("mask", "box"):
            opt = replace(MSSLOptions(), support=support)
            pred = (self.target + torch.tensor([2., -3., 1., -2.])).requires_grad_()
            a = old.side_losses(pred, self.target, self.mask, opt)
            b = side_losses(pred, self.target, self.mask, opt)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            ga = torch.autograd.grad(a.sum(), pred, retain_graph=True)[0]
            gb = torch.autograd.grad(b.sum(), pred)[0]
            torch.testing.assert_close(ga, gb, rtol=0, atol=0)

    def test_parameter_gradient_probe_does_not_change_rng_or_backward(self):
        model=torch.nn.Module(); model.bbox_embed=torch.nn.Linear(2,4)
        boxes=model.bbox_embed(torch.ones(1,2)).sigmoid().reshape(1,1,4)
        native={"loss_bbox":boxes.sum(),"loss_giou":boxes.square().sum()}
        criterion=SimpleNamespace(options=MSSLOptions(),last_diagnostics={"eligible":1},
                                  weight_dict={"loss_bbox":5.,"loss_giou":2.})
        targets=[{"boxes":torch.tensor([[.5,.5,.5,.5]]),"masks":self.mask[None],
                  "mssl_eligible":torch.tensor([True])}]
        indices=[(torch.tensor([0]),torch.tensor([0]))]
        total=5*native["loss_bbox"]+2*native["loss_giou"]
        params=list(model.parameters())
        expected=torch.autograd.grad(total,params,retain_graph=True)
        state=torch.get_rng_state()
        with tempfile.TemporaryDirectory() as directory:
            probe=GradientProbe(model,Path(directory)/"probe.jsonl",275)
            probe.callback(0,0)(criterion,{"pred_boxes":boxes},targets,indices,native)
            self.assertTrue((Path(directory)/"probe.jsonl").exists())
        torch.testing.assert_close(state,torch.get_rng_state(),rtol=0,atol=0)
        actual=torch.autograd.grad(total,params)
        for a,b in zip(actual,expected): torch.testing.assert_close(a,b,rtol=0,atol=0)


if __name__ == "__main__":
    unittest.main()
