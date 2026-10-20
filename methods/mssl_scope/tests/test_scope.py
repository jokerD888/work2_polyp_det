import copy
import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from maskdino.modeling.criterion import SetCriterion
from maskdino.modeling.matcher import HungarianMatcher
from methods.mssl.model import MSSLCriterion
from methods.mssl.config import MSSLOptions
from methods.mssl_scope.model import ScopeCriterion
from methods.mssl_scope.config import validate
from methods.mssl.loss import matched_loss
from methods.mssl_scope.probe import ScopeGradientProbe
from methods.mssl_scope.campaign import smoke_gate


class ScopeTests(unittest.TestCase):
    def test_native_routing_and_gradient_algebra(self):
        kwargs = dict(num_classes=1, matcher=HungarianMatcher(cost_class=4,cost_mask=5,cost_dice=5,
            cost_box=5,cost_giou=2,num_points=32),weight_dict={},eos_coef=.1,losses=['labels','masks','boxes'],
            num_points=32,oversample_ratio=3.,importance_sample_ratio=.75,dn='no',dn_losses=[],
            panoptic_on=False,semantic_ce_loss=False)
        for final_weight, aux_weight in ((1.,0.),(2.,0.),(1.,1.)):
            with self.subTest(weights=(final_weight,aux_weight)):
                torch.manual_seed(42)
                def output():
                    return {'pred_logits':torch.randn(1,6,1,requires_grad=True),
                        'pred_masks':torch.randn(1,6,8,8,requires_grad=True),
                        'pred_boxes':torch.rand(1,6,4,requires_grad=True)*.5+.25}
                outputs=output();outputs['aux_outputs']=[output(),output()];outputs['interm_outputs']=output()
                masks=torch.zeros(1,32,32);masks[:,8:24,8:24]=1
                targets=[{'labels':torch.tensor([0]),'masks':masks,'boxes':torch.tensor([[.5,.5,.5,.5]]),
                          'mssl_eligible':torch.tensor([True])}]
                old=MSSLCriterion(**copy.deepcopy(kwargs));old.options=MSSLOptions()
                new=ScopeCriterion(**copy.deepcopy(kwargs));new.options=MSSLOptions()
                new.final_weight=final_weight;new.aux_weight=aux_weight
                captured={};matches=[]
                new.probe_callback=lambda criterion,losses,parts:captured.update(parts)
                hook=new.matcher.register_forward_hook(lambda module,inputs,result:matches.append((inputs[0],result)))
                state=torch.get_rng_state();expected=old(outputs,targets);after=torch.get_rng_state()
                torch.set_rng_state(state);actual=new(outputs,targets);hook.remove()
                torch.testing.assert_close(torch.get_rng_state(),after,rtol=0,atol=0)
                for key in expected:
                    if key!='loss_mssl':torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
                self.assertEqual(len(matches),4)  # final, two auxiliary, encoder; no extra matcher
                self.assertEqual(new.route_calls,{'final':1,'aux':1})
                aux_match=next(indices for out,indices in matches if out is outputs['aux_outputs'][-1])
                self.assertIs(captured['aux']['indices'],aux_match)
                la,_=matched_loss(outputs['aux_outputs'][-1],targets,aux_match,new.options)
                reference=final_weight*expected['loss_mssl']+aux_weight*la
                torch.testing.assert_close(actual['loss_mssl'],reference,rtol=0,atol=0)
                params=[outputs['pred_boxes'],outputs['aux_outputs'][-1]['pred_boxes']]
                a=torch.autograd.grad(actual['loss_mssl'],params,retain_graph=True,allow_unused=True)
                b=torch.autograd.grad(reference,params,retain_graph=True,allow_unused=True)
                for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
                self.assertEqual(set(old.state_dict()),set(new.state_dict()))

    def test_reject_ambiguous_or_overlapping_protocol(self):
        base={'mssl':{'enabled':True,'weight':1.,'support':'mask'},
              'mssl_scope':{'variant':'F','final_weight':1.,'aux_weight':1.}}
        validate(base)
        for key,value in (('final_weight',.5),('aux_weight',.5)):
            wrong=copy.deepcopy(base);wrong['mssl_scope'][key]=value
            with self.assertRaises(ValueError):validate(wrong)
        wrong=copy.deepcopy(base);wrong['quality_alignment']={}
        with self.assertRaises(ValueError):validate(wrong)

    def test_probe_preserves_rng_grad_and_backward(self):
        torch.manual_seed(123)
        model=torch.nn.Module();model.bbox_embed=torch.nn.Linear(2,2)
        x=model.bbox_embed(torch.ones(1,2))
        lf=x.square().sum();la=(x-1).square().sum();native=(x+2).square().sum()
        losses={'loss_mssl':lf+la,'loss_bbox':native,'loss_bbox_0':native*.5}
        parts={name:{'loss':value,'diagnostics':{'eligible':1},'native_boxes':{'loss_bbox':native}}
               for name,value in (('final',lf),('aux',la))}
        criterion=SimpleNamespace(weight_dict={'loss_mssl':1.,'loss_bbox':5.,'loss_bbox_0':5.},final_weight=1.,aux_weight=1.)
        expected=torch.autograd.grad(lf+la+native,list(model.parameters()),retain_graph=True)
        rng=torch.get_rng_state()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'probe.jsonl'
            ScopeGradientProbe(model,path).callback(0,0)(criterion,losses,parts)
            row=json.loads(path.read_text())
            self.assertLess(row['head_linearity_relative_residual'],1e-6)
            self.assertGreater(row['norms']['total'],0)
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        (lf+la+native).backward()
        for p,g in zip(model.parameters(),expected):torch.testing.assert_close(p.grad,g,rtol=0,atol=0)

    def test_smoke_gate_accepts_finite_zero_diagnostic_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('train','predict_val'):
                (root/name).mkdir();(root/name/'state.json').write_text(json.dumps({'status':'completed'}))
            row={'iteration':49,'optimizer/successful_updates':40,'amp/skipped_step':0,'amp/loss_scale':64.,
                 'total_loss':100.,'loss_mssl':.01,'mssl_scope/weighted_total':.01}
            (root/'train/metrics.json').write_text(json.dumps(row)+'\n')
            probes=[{'iteration':i,'norms':{'total':0.,'native_all':10.}} for i in (0,10)]
            path=root/'train/scope_gradient_probes.jsonl'
            path.write_text(''.join(json.dumps(p)+'\n' for p in probes))
            result=smoke_gate(root,True)
            self.assertEqual(result['zero_norm_probe_iterations'],[0,10])
            probes[0]['norms']['total']=float('nan')
            path.write_text(''.join(json.dumps(p)+'\n' for p in probes))
            with self.assertRaises(RuntimeError):smoke_gate(root,True)


if __name__=='__main__':unittest.main()
