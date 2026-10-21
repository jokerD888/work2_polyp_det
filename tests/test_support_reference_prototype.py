"""CPU-only contracts, not evidence of detector accuracy or trainability."""
import unittest

import torch

from polypbench.analysis.support_reference_prototype import (
    SupportReferencePrototype, _mass_bins, support_descriptor)


class SupportReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.mask = torch.full((1, 3, 10, 12), -4.)
        self.mask[:, 0, 2:8, 3:9] = 4.
        self.mask[:, 1, 4, 4:8] = 4.  # single-row support, bypass
        self.encoder = torch.randn(1, 3, 4)
        self.native = torch.randn(1, 3, 4)
        self.valid = torch.ones(1, 10, 12, dtype=torch.bool)
        self.model = SupportReferencePrototype()

    def args(self):
        return self.mask, self.encoder, self.native, self.valid

    def test_identity_initialization_and_parameter_count(self):
        self.assertTrue(torch.equal(self.model(*self.args()), self.native))
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 5444)

    def test_constructor_does_not_shift_global_cpu_rng(self):
        state = torch.random.get_rng_state().clone()
        SupportReferencePrototype()
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_empty_and_degenerate_stay_native_after_update(self):
        with torch.no_grad():
            self.model.head[-1].bias.fill_(.3)
        out = self.model(*self.args())
        self.assertTrue(torch.equal(out[:, 1:], self.native[:, 1:]))
        self.assertFalse(torch.equal(out[:, :1], self.native[:, :1]))

    def test_mass_integrates_exactly_without_missing_single_pixels(self):
        mass = torch.tensor([[1., 0., 0., 2.]])
        out = _mass_bins(mass, torch.tensor([0]), torch.tensor([3]), 16)
        torch.testing.assert_close(out.sum(-1), torch.ones(1))
        torch.testing.assert_close(out[0, :4].sum(), torch.tensor(1/3))
        torch.testing.assert_close(out[0, 12:].sum(), torch.tensor(2/3))

    def test_profiles_contain_information_beyond_occupancy_and_boxes(self):
        a = torch.full((1, 1, 8, 8), -4.)
        b = a.clone()
        a[0, 0, [0, 0, 7, 7, 1, 1], [0, 7, 0, 7, 1, 6]] = 4.
        b[0, 0, [0, 0, 7, 7, 4, 4], [0, 7, 0, 7, 3, 4]] = 4.
        args = (torch.zeros(1, 1, 4), torch.zeros(1, 1, 4), torch.ones(1, 8, 8, dtype=torch.bool))
        fa, _ = support_descriptor(a, *args)
        fb, _ = support_descriptor(b, *args)
        self.assertTrue(torch.equal(fa[..., 64:], fb[..., 64:]))
        self.assertFalse(torch.equal(fa[..., :64], fb[..., :64]))

    def test_native_encoder_infinite_logits_are_permitted(self):
        self.encoder[:, 0, :2] = torch.tensor([float('inf'), -float('inf')])
        features, _ = support_descriptor(*self.args())
        self.assertTrue(torch.isfinite(features).all())
        self.assertEqual(features[0, 0, -3].item(), 1.)

    def test_padding_foreground_remains_visible(self):
        self.valid[:, :, 7:] = False
        features, _ = support_descriptor(*self.args())
        self.assertGreater(features[0, 0, -6].item(), 0.)

    def test_gradient_reaches_adapter_but_not_detached_sources(self):
        for tensor in (self.mask, self.encoder, self.native):
            tensor.requires_grad_()
        out = self.model(*self.args()).sigmoid()
        loss = (out[:, 0] - torch.tensor([[.3, .4, .2, .2]])).abs().sum()
        loss.backward()
        self.assertGreater(self.model.head[-1].weight.grad.norm().item(), 0.)
        for tensor in (self.mask, self.encoder, self.native):
            self.assertIsNone(tensor.grad)
        # Zero last layer means first-layer gradients start at zero, not broken.
        self.assertEqual(self.model.head[0].weight.grad.norm().item(), 0.)
        with torch.no_grad():
            self.model.head[-1].weight -= .01 * self.model.head[-1].weight.grad
        self.model.zero_grad()
        self.model(*self.args()).sigmoid()[:, 0].sum().backward()
        self.assertGreater(self.model.head[0].weight.grad.norm().item(), 0.)

    def test_correction_is_not_constrained_to_encoder_mask_segment(self):
        self.encoder.zero_()
        self.native.zero_()
        with torch.no_grad():
            self.model.head[-1].bias.copy_(torch.tensor([.2, -.1, .3, -.2]))
        # Any convex interpolation of the two equal input boxes would be .5.
        self.assertFalse(torch.equal(self.model(*self.args()).sigmoid()[:, 0], torch.full((1, 4), .5)))

    def test_nonfinite_mask_is_rejected_without_silent_fallback(self):
        self.mask[0, 0, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            self.model(*self.args())


if __name__ == '__main__':
    unittest.main()
