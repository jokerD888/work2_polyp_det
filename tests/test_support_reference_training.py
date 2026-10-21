import copy
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from methods.support_reference.config import validate
from polypbench.analysis.support_reference_prototype import SupportReferencePrototype, support_descriptor


class TrainingContracts(unittest.TestCase):
    def test_geometry_control_removes_spatial_information_at_equal_stored_parameters(self):
        a = torch.full((1, 1, 8, 8), -4.)
        b = a.clone()
        a[0, 0, [0, 0, 7, 7, 1, 1], [0, 7, 0, 7, 1, 6]] = 4.
        b[0, 0, [0, 0, 7, 7, 4, 4], [0, 7, 0, 7, 3, 4]] = 4.
        args = (torch.zeros(1, 1, 4), torch.zeros(1, 1, 4), torch.ones(1, 8, 8, dtype=torch.bool))
        fa, _ = support_descriptor(a, *args)
        fb, _ = support_descriptor(b, *args)
        index = int(torch.nonzero(fa[0, 0, :64] != fb[0, 0, :64])[0])
        spatial = SupportReferencePrototype()
        geometric = SupportReferencePrototype(use_profiles=False)
        with torch.no_grad():
            for p in spatial.parameters():
                p.zero_()
            spatial.head[0].weight[0, index] = 1.
            spatial.head[0].bias[0] = 1.
            spatial.head[-1].weight[0, 0] = 1.
        geometric.load_state_dict(spatial.state_dict(), strict=True)
        self.assertFalse(torch.equal(spatial(a, *args), spatial(b, *args)))
        self.assertTrue(torch.equal(geometric(a, *args), geometric(b, *args)))
        self.assertEqual(sum(p.numel() for p in spatial.parameters()),
                         sum(p.numel() for p in geometric.parameters()))

    def test_fixed_variants_reject_coefficient_search_and_route_overlap(self):
        example = {'support_reference': {'variant': 'support'},
            'runtime': {'num_gpus': 1, 'micro_batch': 1}, 'mssl': {'enabled': False}}
        for variant in ('native', 'blend05', 'geometry', 'support'):
            valid = copy.deepcopy(example)
            valid['support_reference']['variant'] = variant
            validate(valid)
        invalid = copy.deepcopy(example)
        invalid['support_reference']['coefficient'] = .4
        with self.assertRaises(ValueError):
            validate(invalid)
        invalid = copy.deepcopy(example)
        invalid['quality_alignment'] = {}
        with self.assertRaises(ValueError):
            validate(invalid)

    def test_gt_events_join_annotation_ids_and_keep_negative_frames(self):
        from polypbench.io import write_json, read_json
        from methods.support_reference.comparison import run
        from polypbench.zeroshot import score_cohort
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gt = {'images': [{'id': i, 'width': 100, 'height': 100, 'file_name': str(i)} for i in (1,2)],
                'categories': [{'id': 1, 'name': 'polyp'}], 'annotations': [
                    {'id': 11, 'image_id': 1, 'category_id': 1, 'bbox': [0,0,10,10], 'area': 100, 'iscrowd': 0},
                    {'id': 12, 'image_id': 1, 'category_id': 1, 'bbox': [30,30,10,10], 'area': 100, 'iscrowd': 0}]}
            write_json(root/'gt.json', gt)
            stages = root/'native'
            write_json(stages/'images/1.json', {'annotation_ids': [12,11], 'center': 'C1',
                'stages': {'decoder_09': {'max_iou': [.1,.8]}}})
            write_json(stages/'images/2.json', {'annotation_ids': [], 'center': 'C1',
                'stages': {'decoder_09': {'max_iou': []}}})
            write_json(root/'protocol.json', {'annotations': str(root/'gt.json'), 'native_stages': str(stages),
                'reference_metrics': {'baseline': {'AP': .4}}})
            for variant in ('blend05', 'geometry', 'support'):
                external = root/'external'/variant
                (external/'candidates').mkdir(parents=True)
                predictions = [{'image_id': i, 'category_id': 1, 'bbox': [30,30,10,10], 'score': .9} for i in (1,2)]
                write_json(external/'predictions.all_stills.bbox.json', predictions)
                for i in (1,2):
                    arrays = {name+'_boxes_xyxy': np.array([[30,30,40,40.]]) for name in
                              ('encoder','mask_native','initial','final300')}
                    arrays['top100_boxes_xyxy'] = arrays['final300_boxes_xyxy']
                    np.savez(external/'candidates'/f'{i}.npz', **arrays)
                with contextlib.redirect_stdout(io.StringIO()):
                    score_cohort(root/'gt.json', predictions, external/'primary')
            with contextlib.redirect_stdout(io.StringIO()):
                run(root)
            comparison = read_json(root/'comparison.json')
            for condition in comparison['conditions'].values():
                self.assertEqual(condition['marginal_gt_events']['all']['0.5'],
                                 {'recovered': 1, 'newly_lost': 1, 'net': 0})
                self.assertEqual(condition['negative_frames']['negative_frames_with_detection'], 1)


if __name__ == '__main__':
    unittest.main()
