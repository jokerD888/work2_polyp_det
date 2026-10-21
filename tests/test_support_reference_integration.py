"""Image-only geometry and explicit state-dict acceptance contracts."""
import unittest
from types import SimpleNamespace
import torch
from torch import nn
from methods.support_reference.integration import valid_canvas, load_native_checkpoint
from polypbench.analysis.support_reference_acceptance import tensor_digest, panel_from_annotations, gradient_groups


class IntegrationContracts(unittest.TestCase):
    def test_adapter_layer_norm_does_not_include_native_parameter_gradients(self):
        adapter = nn.Sequential(nn.Linear(1, 1), nn.Linear(1, 1))
        native_box = nn.Parameter(torch.zeros(1))
        native_mask = nn.Parameter(torch.zeros(1))
        layer_box = SimpleNamespace(bias=native_box)
        layer_mask = SimpleNamespace(bias=native_mask)
        predictor = SimpleNamespace(_bbox_embed=SimpleNamespace(layers=[layer_box]),
                                    mask_embed=SimpleNamespace(layers=[layer_mask]))
        model = SimpleNamespace(sem_seg_head=SimpleNamespace(predictor=predictor),
            parameters=lambda: iter(list(adapter.parameters()) + [native_box, native_mask]))
        active = sum(p.sum() for p in adapter.parameters())
        losses = {'loss_bbox_0': active, 'loss_giou_0': active * 0,
                  'loss_bbox_1': active + 100 * native_box.sum(),
                  'loss_bbox_interm': native_box.sum()}
        rows = gradient_groups(model, SimpleNamespace(adapter=adapter), losses)
        self.assertAlmostEqual(rows['first_decoder_box']['adapter_last_layer_l2'], 2**.5, places=5)
        self.assertAlmostEqual(rows['first_decoder_box']['native_bbox_last_bias_l2'], 100.)
    def test_native_eval_optional_auxiliary_masks_are_valid(self):
        self.assertIsNone(tensor_digest(None))

    def test_sparse_annotation_strata_are_not_filled_with_duplicates(self):
        gt = {'images': [{'id': i, 'width': 100, 'height': 100, 'file_name': str(i)} for i in range(3)],
              'annotations': [{'image_id': i, 'bbox': [0, 0, 10+i, 10+i]} for i in range(3)]}
        gt['annotations'].append({'image_id': 0, 'bbox': [20, 20, 10, 10]})
        panel = panel_from_annotations(gt)
        self.assertEqual(len(panel['images']), 3)
        self.assertEqual(len({i['id'] for i in panel['images']}), 3)
        self.assertFalse(panel['population_prevalence_estimate'])

    def test_mapper_and_batch_padding_are_distinct(self):
        padding = torch.zeros(7, 9, dtype=torch.bool)
        padding[4:] = True
        inputs = [{'image': torch.zeros(3, 7, 9), 'padding_mask': padding,
                   'instances': 'must never be inspected'},
                  {'image': torch.zeros(3, 5, 6)}]
        canvas = valid_canvas(inputs, 8, 'cpu')
        self.assertEqual(tuple(canvas.shape), (2, 8, 16))
        self.assertEqual(canvas[0].sum().item(), 4 * 9)
        self.assertEqual(canvas[1].sum().item(), 5 * 6)

    def test_invalid_mapper_padding_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Mapper padding'):
            valid_canvas([{'image': torch.zeros(3, 3, 4),
                           'padding_mask': torch.zeros(3, 4)}], 8, 'cpu')

    def test_only_adapter_weights_may_be_missing(self):
        model = nn.Module()
        model.detector = nn.Module()
        model.detector.base = nn.Linear(2, 2)
        state = {k: v.clone() for k, v in model.state_dict().items()}
        model.detector.support_reference = nn.Linear(2, 4)
        self.assertEqual(set(load_native_checkpoint(model, state)),
            {'detector.support_reference.weight', 'detector.support_reference.bias'})
        bad = dict(state)
        del bad['detector.base.bias']
        with self.assertRaisesRegex(RuntimeError, 'Unexpected checkpoint'):
            load_native_checkpoint(model, bad)
        bad = dict(state, typo=torch.ones(1))
        with self.assertRaisesRegex(RuntimeError, 'Unexpected checkpoint'):
            load_native_checkpoint(model, bad)


if __name__ == '__main__':
    unittest.main()
