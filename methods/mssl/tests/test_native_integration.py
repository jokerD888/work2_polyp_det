import copy
import unittest
import torch

from maskdino.modeling.criterion import SetCriterion
from maskdino.modeling.matcher import HungarianMatcher
from methods.mssl.model import MSSLCriterion
from methods.mssl.config import MSSLOptions


class IntegrationTests(unittest.TestCase):
    def test_native_losses_rng_and_matching_unchanged(self):
        kwargs = dict(num_classes=1,matcher=HungarianMatcher(cost_class=4,cost_mask=5,
            cost_dice=5,cost_box=5,cost_giou=2,num_points=32),weight_dict={},eos_coef=.1,
            losses=["labels","masks","boxes"],num_points=32,oversample_ratio=3.,
            importance_sample_ratio=.75,dn="no",dn_losses=[],panoptic_on=False,
            semantic_ce_loss=False)
        native = SetCriterion(**kwargs)
        method = MSSLCriterion(**copy.deepcopy(kwargs))
        method.options = MSSLOptions()
        torch.manual_seed(42)
        def output():
            return {"pred_logits":torch.randn(1,6,1,requires_grad=True),
                    "pred_masks":torch.randn(1,6,8,8,requires_grad=True),
                    "pred_boxes":torch.rand(1,6,4,requires_grad=True)*.5+.25}
        outputs=output();outputs["aux_outputs"]=[output()];outputs["interm_outputs"]=output()
        mask=torch.zeros(1,32,32);mask[:,8:24,8:24]=1
        targets=[{"labels":torch.tensor([0]),"masks":mask,
                  "boxes":torch.tensor([[.5,.5,.5,.5]]),"mssl_eligible":torch.tensor([True])}]
        state=torch.get_rng_state()
        expected=native(outputs,targets)
        expected_rng=torch.get_rng_state()
        torch.set_rng_state(state)
        actual=method(outputs,targets)
        self.assertEqual(set(actual)-set(expected),{"loss_mssl"})
        torch.testing.assert_close(torch.get_rng_state(),expected_rng,rtol=0,atol=0)
        for key in expected:
            torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
        grad=torch.autograd.grad(actual["loss_mssl"],outputs["pred_boxes"])[0]
        self.assertTrue(torch.isfinite(grad).all() and grad.abs().sum()>0)
        self.assertEqual(set(method.state_dict()),set(native.state_dict()))


if __name__=="__main__":
    unittest.main()
