import unittest
from dataclasses import replace

import torch

from methods.mssl.config import MSSLOptions
from methods.mssl.loss import side_losses, matched_loss


class LossTests(unittest.TestCase):
    def setUp(self):
        self.options = MSSLOptions()
        self.mask = torch.zeros(32, 32)
        self.mask[5:27, 7:25] = 1
        self.target = torch.tensor([7., 5., 25., 27.])

    def test_zero_at_provided_box_even_with_mask_margin(self):
        for target in (self.target, torch.tensor([3., 2., 30., 31.])):
            pred = target.clone().requires_grad_()
            loss = side_losses(pred, target, self.mask, self.options).sum()
            loss.backward()
            self.assertEqual(loss.item(), 0.)
            torch.testing.assert_close(pred.grad, torch.zeros(4))

    def test_directions_and_finite_difference(self):
        for direction in (-1., 1.):
            pred = (self.target + direction*.6).requires_grad_()
            loss = side_losses(pred, self.target, self.mask, self.options).sum()
            loss.backward()
            self.assertTrue((direction*pred.grad > 0).all())
            for i in range(4):
                plus, minus = pred.detach().clone(), pred.detach().clone()
                plus[i] += .002; minus[i] -= .002
                fd = (side_losses(plus,self.target,self.mask,self.options).sum() -
                      side_losses(minus,self.target,self.mask,self.options).sum()) / .004
                torch.testing.assert_close(fd,pred.grad[i],rtol=.01,atol=1e-5)

    def test_mask_increment_same_box(self):
        mask = self.mask.clone()
        mask[6:26,8:24] = 0
        pred = self.target + torch.tensor([2., 2., -2., -2.])
        self.assertFalse(torch.allclose(side_losses(pred,self.target,self.mask,self.options),
                                        side_losses(pred,self.target,mask,self.options)))
        box = replace(self.options,support="box")
        torch.testing.assert_close(side_losses(pred,self.target,self.mask,box),
                                   side_losses(pred,self.target,mask,box))

    def test_empty_support_and_no_gt_have_finite_graph_zero(self):
        self.assertIsNone(side_losses(self.target,self.target,torch.zeros(32,32),self.options))
        pred = torch.rand(1,3,4,requires_grad=True)
        loss,diag = matched_loss({"pred_boxes":pred},[{"masks":torch.empty(0,32,32),
            "mssl_eligible":torch.empty(0,dtype=torch.bool),"boxes":torch.empty(0,4)}],
            [(torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long))],self.options)
        loss.backward()
        self.assertEqual(diag["eligible"],0)
        torch.testing.assert_close(pred.grad,torch.zeros_like(pred))

    def test_multi_original_ineligible(self):
        pred=torch.tensor([[[.45,.45,.5,.5]]],requires_grad=True)
        target={"masks":self.mask[None],"boxes":torch.tensor([[.5,.5,.5,.5]]),
                "mssl_eligible":torch.tensor([False])}
        loss,diag=matched_loss({"pred_boxes":pred},[target],[(torch.tensor([0]),torch.tensor([0]))],self.options)
        self.assertEqual(diag["eligible"],0)
        self.assertEqual(loss.item(),0.)

    def test_invalid_options_fail(self):
        for key,value in (("temperature",0),("weight",-1),("support","predicted_mask")):
            with self.assertRaises(ValueError):
                replace(self.options,**{key:value}).validate()


if __name__=="__main__":
    unittest.main()
