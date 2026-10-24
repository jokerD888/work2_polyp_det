import copy
from dataclasses import replace
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from methods.local_contrast.config import Options
from methods.local_contrast.loss import supervised_contrast, region_loss
from methods.local_contrast.regions import plan, occupancy


def record(multi=False, tiny=False):
    size = 128
    boxes = [[32, 32, 80, 80]] if not multi else [[16, 16, 56, 56], [72, 72, 112, 112]]
    if tiny:
        boxes = [[33, 33, 35, 35]]
    masks = torch.zeros(len(boxes), size, size, dtype=torch.bool)
    for i, (a, b, c, d) in enumerate(boxes):
        masks[i, b:d, a:c] = True
    return dict(image=torch.zeros(3, size, size), image_id=123,
        padding_mask=torch.zeros(size, size, dtype=torch.bool), local_semantic_mask=masks.any(0),
        instances=SimpleNamespace(gt_boxes=SimpleNamespace(tensor=torch.tensor(boxes).float()),
            gt_masks=masks, gt_mssl_eligible=torch.tensor([not multi]*len(boxes))))


class LocalTests(unittest.TestCase):
    def test_loss_finite_gradient_and_permutation(self):
        z = torch.randn(8, 6, requires_grad=True)
        loss = supervised_contrast(z, 4, .1)
        g = torch.autograd.grad(loss, z)[0]
        self.assertTrue(torch.isfinite(g).all() and g.abs().sum() > 0)
        perm = torch.tensor([2, 0, 3, 1, 6, 5, 4, 7])
        torch.testing.assert_close(loss, supervised_contrast(z[perm], 4, .1))

    def test_separation_reduces_loss(self):
        mixed = torch.ones(8, 2)
        separated = torch.tensor([[1., 0.]]*4+[[0., 1.]]*4)
        self.assertLess(supervised_contrast(separated, 4, .1), supervised_contrast(mixed, 4, .1))

    def test_loss_matches_explicit_definition(self):
        z = torch.nn.functional.normalize(torch.randn(7, 5), dim=1)
        actual = supervised_contrast(z, 3, .2)
        anchors = []
        for i in range(7):
            positives = [j for j in range(7) if j != i and (j < 3) == (i < 3)]
            denominator = sum(torch.exp(z[i]@z[k]/.2) for k in range(7) if k != i)
            anchors.append(sum(-torch.log(torch.exp(z[i]@z[j]/.2)/denominator) for j in positives)/len(positives))
        expected = (torch.stack(anchors[:3]).mean()+torch.stack(anchors[3:]).mean())/2
        torch.testing.assert_close(actual, expected)

    def test_paired_counts_unique_and_rng(self):
        row = record()
        row['instances'].gt_masks[0, 32:48, 32:80] = False
        state = torch.get_rng_state().clone()
        state_np = np.random.get_state()
        a, sa = plan(row, Options())
        b, sb = plan(row, Options(variant='mask_priority'))
        self.assertTrue(a and b)
        self.assertEqual(sa['used'], sb['used'])
        self.assertEqual([(len(r['fg']), len(r['bg'])) for r in a], [(len(r['fg']), len(r['bg'])) for r in b])
        self.assertEqual(b[0]['source'], 'mask')
        for rows in (a, b):
            for r in rows:
                self.assertEqual(len(set(r['fg']) | set(r['bg'])), len(r['fg'])+len(r['bg']))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertTrue(np.array_equal(state_np[1], np.random.get_state()[1]))

    def test_multibox_identical_fallback(self):
        row = record(multi=True)
        a, sa = plan(row, Options())
        b, sb = plan(row, Options(variant='mask_priority'))
        self.assertEqual(sa, sb)
        self.assertEqual(sa['used'], 2)
        self.assertEqual(sa['box_fallback'], 2)
        for ar, br in zip(a, b):
            np.testing.assert_array_equal(ar['fg'], br['fg'])
            np.testing.assert_array_equal(ar['bg'], br['bg'])

    def test_empty_mask_falls_back_in_both(self):
        row = record()
        row['instances'].gt_masks.zero_()
        a, sa = plan(row, Options())
        b, sb = plan(row, Options(variant='mask_priority'))
        self.assertEqual(sa, sb)
        self.assertEqual(sa['box_fallback'], 1)
        np.testing.assert_array_equal(a[0]['fg'], b[0]['fg'])

    def test_tiny_common_skip_and_differentiable_zero(self):
        a, sa = plan(record(tiny=True), Options())
        b, sb = plan(record(tiny=True), Options(variant='mask_priority'))
        self.assertEqual(sa, sb)
        self.assertFalse(a or b)
        self.assertEqual(sa['skipped'], 1)
        grid = torch.randn(8, 16, 16, requires_grad=True)
        loss = region_loss(grid, [], .1)
        self.assertEqual(float(loss), 0.)
        self.assertTrue(torch.equal(torch.autograd.grad(loss, grid)[0], torch.zeros_like(grid)))

    def test_padding_and_semantic_excluded_from_background(self):
        row = record(multi=True)
        row['padding_mask'][:, :8] = True
        row['local_semantic_mask'][48:64, 8:64] = True
        rows, _ = plan(row, Options())
        forbidden = occupancy(row['local_semantic_mask'].numpy() | row['padding_mask'].numpy()) > .05
        for r in rows:
            self.assertFalse(forbidden.flatten()[r['bg']].any())

    def test_no_foreground_augmentation_labels_changed(self):
        row = record()
        before = copy.deepcopy(row)
        plan(row, Options())
        self.assertTrue(torch.equal(before['instances'].gt_masks, row['instances'].gt_masks))
        self.assertTrue(torch.equal(before['instances'].gt_boxes.tensor, row['instances'].gt_boxes.tensor))

    def test_options_reject_invalid(self):
        for kwargs in (dict(temperature=0), dict(points=1), dict(purity=1.1), dict(weight=float('nan'))):
            with self.assertRaises(ValueError):
                Options(**kwargs).validate()

    def test_external_comparison_misses_negatives_groups_and_deltas(self):
        from methods.local_contrast.comparison import run
        from polypbench.io import write_json, read_json
        gt = dict(images=[dict(id=1, width=128, height=128, file_name='1.jpg'),
                          dict(id=2, width=128, height=128, file_name='2.jpg')],
            categories=[dict(id=1, name='polyp')],
            annotations=[dict(id=1, image_id=1, category_id=1, bbox=[32, 32, 32, 32], area=1024, iscrowd=0)])
        predictions = [dict(image_id=1, category_id=1, bbox=[32, 32, 32, 32], score=.9),
                       dict(image_id=2, category_id=1, bbox=[8, 8, 16, 16], score=.6)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root/'gt.json', gt)
            for c in range(1, 7):
                write_json(root/f'C{c}.coco.json', gt)
            write_json(root/'mssl.json', predictions)
            for v in ('box', 'mask_priority'):
                write_json(root/'external'/v/'predictions.all_stills.bbox.json', predictions)
            write_json(root/'protocol.json', dict(annotations=str(root/'gt.json'), mssl_predictions=str(root/'mssl.json')))
            run(root)
            result = read_json(root/'comparison.json')
            self.assertEqual(result['deltas']['mask_priority']['box']['AP'], 0)
            self.assertEqual(result['conditions']['box']['matched_recall_at_100']['0.50']['misses'], 0)
            self.assertEqual(result['conditions']['box']['negative_frames_at_score05']['fp'], 1)
            self.assertIn('C6', result['conditions']['box']['groups'])


if __name__ == '__main__':
    unittest.main()
