import copy
from types import SimpleNamespace
import unittest
import torch
from torch import nn
from methods.reference_features.integration import ExtraRead, ReferenceFeaturesAttachment, scaled_encoder_reference
from methods.reference_features.config import validate


class FakeSampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(4))

    def forward(self, query, reference, memory, shapes, starts, padding):
        return (query + reference.mean(2)) @ self.weight


class FeatureContracts(unittest.TestCase):
    def test_constructor_preserves_rng_and_zero_output(self):
        sampler = FakeSampler()
        before = torch.random.get_rng_state().clone()
        extra = ExtraRead(sampler, 4)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        q = torch.randn(2, 300, 4)
        self.assertTrue(torch.equal(extra(q, torch.rand(2, 300, 3, 4), q, None, None, None), torch.zeros_like(q)))
        self.assertIsNot(extra.sampler.weight, sampler.weight)

    def test_zero_output_learns_then_sampler_receives_gradient(self):
        extra = ExtraRead(FakeSampler(), 4)
        opt = torch.optim.SGD(extra.parameters(), lr=.01)
        q, r = torch.randn(2, 300, 4), torch.rand(2, 300, 3, 4)
        extra(q, r, q, None, None, None).sum().backward()
        self.assertGreater(extra.projection.weight.grad.norm().item(), 0)
        self.assertEqual(extra.sampler.weight.grad.norm().item(), 0)
        opt.step()
        opt.zero_grad()
        extra(q, r, q, None, None, None).sum().backward()
        self.assertGreater(extra.sampler.weight.grad.norm().item(), 0)

    def test_valid_ratios_and_batch_coordinates(self):
        logits = torch.zeros(2, 300, 4)
        ratios = torch.tensor([[[1., .5], [.8, .4]], [[.3, .7], [.6, 1.]]])
        actual = scaled_encoder_reference(logits, ratios)
        self.assertEqual(actual.shape, (2, 300, 2, 4))
        torch.testing.assert_close(actual[:, 0], .5*torch.cat([ratios, ratios], -1))
        logits.requires_grad_()
        self.assertFalse(scaled_encoder_reference(logits, ratios).requires_grad)

    def attachment(self, variant):
        a = ReferenceFeaturesAttachment(SimpleNamespace(sem_seg_head=SimpleNamespace(predictor=None)), variant, capture=True)
        a.adapter = ExtraRead(FakeSampler(), 4)
        with torch.no_grad():
            a.adapter.projection.weight.copy_(torch.eye(4))
        a.context = torch.ones(2, 300, 2, 4)*.7
        return a

    def test_dn_prefix_unchanged_and_no_direct_branch_gradient(self):
        a = self.attachment('encoder_extra')
        q = torch.randn(2, 305, 4)
        refs = torch.zeros(2, 305, 2, 4)
        out = q.clone()
        result = a._extra(None, (q, refs, q, None, None, None), out)
        self.assertTrue(torch.equal(result[:, :5], out[:, :5]))
        self.assertFalse(torch.equal(result[:, 5:], out[:, 5:]))
        result[:, :5].sum().backward()
        self.assertEqual(a.adapter.projection.weight.grad.norm().item(), 0)
        self.assertTrue(a.last['dn_cross_unchanged'])

    def test_only_extra_read_reference_differs(self):
        a, b = self.attachment('mask_extra'), self.attachment('encoder_extra')
        b.adapter.load_state_dict(a.adapter.state_dict())
        q = torch.randn(2, 300, 4)
        refs = torch.ones(2, 300, 2, 4)*.2
        inputs = (q, refs, q, None, None, None)
        main_before = refs.clone()
        ra, rb = a._extra(None, inputs, q), b._extra(None, inputs, q)
        self.assertFalse(torch.equal(ra, rb))
        self.assertTrue(torch.equal(refs, main_before))
        self.assertTrue(torch.equal(a.last['main_reference'], b.last['main_reference']))
        self.assertEqual(sum(p.numel() for p in a.adapter.parameters()), sum(p.numel() for p in b.adapter.parameters()))

    def test_disabled_branch_is_identity(self):
        a = self.attachment('encoder_extra')
        a.enabled = False
        q = torch.randn(2, 300, 4)
        self.assertIs(a._extra(None, (), q), q)

    def test_reject_tuning_and_overlapping_routes(self):
        base = {'reference_features': {'variant': 'encoder_extra'},
            'runtime': {'num_gpus': 1, 'micro_batch': 1}, 'mssl': {'enabled': False}}
        validate(base)
        for key, value in [('support_reference', {}), ('quality_alignment', {})]:
            bad = dict(base, **{key: value})
            with self.assertRaises(ValueError):
                validate(bad)
        bad = copy.deepcopy(base)
        bad['reference_features']['layers'] = [0, 1]
        with self.assertRaises(ValueError):
            validate(bad)

    def test_batch_loss_rounding_never_relaxes_prediction_identity(self):
        from methods.reference_features.acceptance import compare_batch_summary
        expected = {'boxes': 'native', 'losses': {'loss_mask': float(torch.tensor(2.7))}}
        actual = copy.deepcopy(expected)
        actual['losses']['loss_mask'] = float(torch.nextafter(torch.tensor(2.7), torch.tensor(float('inf'))))
        compare_batch_summary(expected, actual)
        actual['boxes'] = 'changed'
        with self.assertRaises(AssertionError):
            compare_batch_summary(expected, actual)
        actual = copy.deepcopy(expected)
        actual['losses']['loss_mask'] += .001
        with self.assertRaises(AssertionError):
            compare_batch_summary(expected, actual)


if __name__ == '__main__':
    unittest.main()
