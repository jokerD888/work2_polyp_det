import copy
import unittest
from dataclasses import replace
import torch

from maskdino.modeling.criterion import SetCriterion
from maskdino.modeling.matcher import HungarianMatcher
from methods.mssl.model import MSSLCriterion
from methods.mssl.config import MSSLOptions
from methods.quality_alignment.model import QualityCriterion


class RoutingTests(unittest.TestCase):
    def test_only_final_classification_changes_with_and_without_mssl(self):
        kwargs = dict(num_classes=1, matcher=HungarianMatcher(cost_class=4,cost_mask=5,
            cost_dice=5,cost_box=5,cost_giou=2,num_points=32),weight_dict={},eos_coef=.1,
            losses=['labels','masks','boxes'],num_points=32,oversample_ratio=3.,
            importance_sample_ratio=.75,dn='no',dn_losses=[],panoptic_on=False,semantic_ce_loss=False)
        for enabled in (False, True):
            native = MSSLCriterion(**copy.deepcopy(kwargs)) if enabled else SetCriterion(**copy.deepcopy(kwargs))
            options = replace(MSSLOptions(), enabled=enabled)
            if enabled: native.options = options
            method = QualityCriterion(**copy.deepcopy(kwargs)); method.mssl_options = options
            torch.manual_seed(42)
            def output():
                return {'pred_logits':torch.randn(1,6,1,requires_grad=True),
                        'pred_masks':torch.randn(1,6,8,8,requires_grad=True),
                        'pred_boxes':torch.rand(1,6,4,requires_grad=True)*.5+.25}
            outputs=output();outputs['aux_outputs']=[output(),output()];outputs['interm_outputs']=output()
            mask=torch.zeros(1,32,32);mask[:,8:24,8:24]=1
            targets=[{'labels':torch.tensor([0]),'masks':mask,
                      'boxes':torch.tensor([[.5,.5,.5,.5]]),'mssl_eligible':torch.tensor([True])}]
            state=torch.get_rng_state();expected=native(outputs,targets);after=torch.get_rng_state()
            torch.set_rng_state(state);actual=method(outputs,targets)
            self.assertEqual(set(actual),set(expected));self.assertEqual(method._q_calls,1)
            torch.testing.assert_close(torch.get_rng_state(),after,rtol=0,atol=0)
            self.assertFalse(torch.equal(actual['loss_ce'],expected['loss_ce']))
            for key in expected:
                if key!='loss_ce': torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
            if enabled:
                ga=torch.autograd.grad(actual['loss_mssl'],outputs['pred_boxes'],retain_graph=True)[0]
                ge=torch.autograd.grad(expected['loss_mssl'],outputs['pred_boxes'],retain_graph=True)[0]
                torch.testing.assert_close(ga,ge,rtol=0,atol=0)
            self.assertEqual(set(method.state_dict()),set(native.state_dict()))


if __name__=='__main__': unittest.main()
