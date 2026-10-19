import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import torch
from methods.quality_alignment.loss import quality_loss
from methods.quality_alignment.probe import QualityGradientProbe


class ProbeTests(unittest.TestCase):
    def test_predetermined_measurement_backoff_handles_half_precision_overflow(self):
        model=torch.nn.Module();model.class_embed=torch.nn.Linear(3,1).half()
        with torch.no_grad():model.class_embed.weight.zero_();model.class_embed.bias.zero_()
        logits=model.class_embed(torch.ones(1,1000,3,dtype=torch.float16))
        empty=torch.empty(0,dtype=torch.long)
        parts=quality_loss(logits,torch.empty(0,4),torch.empty(0,4),empty,empty,empty,1.)
        parts['native']=torch.nn.functional.softplus(logits.float()).mean()
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'probe.jsonl';probe=QualityGradientProbe(model,file,275)
            probe.callback(0,0)(SimpleNamespace(weight_dict={'loss_ce':4}),{'pred_logits':logits},None,None,parts)
            record=json.loads(file.read_text())['gradients']['loss']
        self.assertFalse(record['attempts'][0]['finite'])
        self.assertEqual(record['measurement_scale'],8.)
        self.assertGreater(record['class_head_gradient_norm'],0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_probe_preserves_rng_grad_and_final_backward(self):
        torch.manual_seed(7)
        model=torch.nn.Module();model.class_embed=torch.nn.Linear(3,1)
        logits=model.class_embed(torch.randn(1,4,3))
        box=torch.tensor([[.5,.5,.2,.2]]);idx=torch.tensor([0])
        parts=quality_loss(logits,box,box,idx,idx,idx,1.)
        parts['native']=torch.nn.functional.softplus(logits).mean()
        params=list(model.parameters())
        expected=torch.autograd.grad(parts['loss']*4,params,retain_graph=True)
        state=torch.get_rng_state()
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'probe.jsonl';probe=QualityGradientProbe(model,file,275)
            probe.callback(0,0)(SimpleNamespace(weight_dict={'loss_ce':4}),{'pred_logits':logits},None,None,parts)
            row=json.loads(file.read_text());self.assertEqual(row['gradients']['loss']['measurement_scale'],64.)
            self.assertIsNone(probe.callback(1,0));self.assertIsNone(probe.callback(0,1))
        torch.testing.assert_close(torch.get_rng_state(),state,rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in params))
        (parts['loss']*4).backward()
        for parameter,gradient in zip(params,expected):torch.testing.assert_close(parameter.grad,gradient,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
