"""Validate external collector on two source images per passed smoke model."""
import argparse
import copy
import gc
from collections import Counter
from pathlib import Path
import sys
import numpy as np
from polypbench.io import read_json, write_json, sha256


def run(smoke, output):
    from methods.mssl.config import load_experiment, build_cfg
    from .external import run as external_run
    p = read_json(smoke/'protocol.json')
    assert read_json(smoke/'verification.json')['status'] == 'passed'
    output.mkdir(parents=True, exist_ok=False)
    receipts = []
    for variant in ('blend05', 'geometry', 'support'):
        experiment = load_experiment(p['configs'][variant])
        experiment['dataset']['val'] = copy.deepcopy(experiment['dataset']['train'])
        sys.path.insert(0, experiment['repo']['path'])
        import torch
        import train_net  # noqa: F401
        from detectron2.modeling import build_model
        from detectron2.data import DatasetMapper
        from detectron2.evaluation.coco_evaluation import instances_to_coco_json
        gt = read_json(experiment['dataset']['train']['annotations'])
        counts = Counter(a['image_id'] for a in gt['annotations'])
        first = gt['images'][0]
        second = next(im for im in gt['images'] if (counts[im['id']] > 1) != (counts[first['id']] > 1))
        images = [first, second]
        directory = output/variant
        directory.mkdir()
        write_json(directory/'experiment.json', experiment)
        cfg = build_cfg(experiment, directory, device='cuda')
        model = build_model(cfg)
        checkpoint = smoke/variant/'smoke_only.pth'
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'], strict=True)
        del state
        model.eval()
        mapper = DatasetMapper(cfg, False)
        expected = []
        for image in images:
            record = dict(image)
            record['file_name'] = str(Path(experiment['dataset']['train']['image_root'])/record['file_name'])
            record['image_id'] = record.pop('id')
            mapped = mapper(record)
            assert 'annotations' not in mapped and 'instances' not in mapped
            with torch.no_grad():
                instances = model([mapped])[0]['instances']
            instances.remove('pred_masks')
            rows = instances_to_coco_json(instances.to('cpu'), image['id'])
            for row in rows:
                row['category_id'] = 1
            expected.extend(rows)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        write_json(directory/'plain_predictions.json', expected)
        write_json(directory/'images.json', {'images': images})
        write_json(directory/'job.json', {'experiment': str(directory/'experiment.json'),
            'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint),
            'output': str(directory/'observed'), 'image_root': experiment['dataset']['train']['image_root'],
            'all_images': str(directory/'images.json'), 'source_predictions': str(directory/'plain_predictions.json')})
        external_run(directory/'job.json')
        gc.collect()
        actual = read_json(directory/'observed/predictions.all_stills.bbox.json')
        assert len(actual) == len(expected)
        np.testing.assert_array_equal([r['bbox'] for r in actual], [r['bbox'] for r in expected])
        np.testing.assert_array_equal([r['score'] for r in actual], [r['score'] for r in expected])
        for image in images:
            with np.load(directory/'observed/candidates'/f"{image['id']}.npz") as arrays:
                for name in ('encoder', 'mask_native', 'initial', 'final300'):
                    assert arrays[name+'_boxes_xyxy'].shape == (300, 4)
        receipts.append({'variant': variant, 'images': [im['id'] for im in images],
            'strict_smoke_checkpoint': True, 'observed_vs_plain_predictions_bitwise': True,
            'four_stage_candidate_shapes': True})
    write_json(output/'verification.json', {'status': 'passed', 'receipts': receipts,
        'source_train_only': True, 'ap_evaluation': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(Path(args.smoke).resolve(), Path(args.output).resolve())
