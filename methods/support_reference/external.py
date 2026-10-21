"""Image-only external inference with small per-query geometry exports."""
import argparse
from pathlib import Path
import sys
import time
import numpy as np
from polypbench.io import read_json, write_json, sha256


def run(job_path):
    job = read_json(job_path)
    experiment = read_json(job['experiment'])
    sys.path.insert(0, experiment['repo']['path'])
    import torch
    import train_net  # noqa: F401
    from detectron2.modeling import build_model
    from detectron2.data import DatasetMapper
    from detectron2.evaluation.coco_evaluation import instances_to_coco_json
    from methods.mssl.config import build_cfg
    from polypbench.analysis.candidate_stage import ReturnObserver, to_original, cpu
    assert sha256(job['checkpoint']) == job['checkpoint_sha256']
    output = Path(job['output'])
    output.mkdir(parents=True, exist_ok=True)
    cfg = build_cfg(experiment, output)
    model = build_model(cfg)
    saved = torch.load(job['checkpoint'], map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'], strict=True)
    del saved
    model.eval()
    mapper = DatasetMapper(cfg, False)
    diagnostics = output/'candidates'
    diagnostics.mkdir(exist_ok=False)
    versions = {k: p._version for k, p in model.state_dict(keep_vars=True).items()}

    def predict(image, image_root, export=False):
        record = dict(image)
        record['file_name'] = str(Path(image_root)/image['file_name'])
        record['image_id'] = record.pop('id')
        mapped = mapper(record)
        assert 'instances' not in mapped and 'annotations' not in mapped
        data = {}
        decoder = model.sem_seg_head.predictor
        attachment = model.reference_attachment
        attachment.capture = export
        def decoder_return(loc, result):
            if result is None:
                return
            out, known = result
            assert known is None and out['pred_boxes'].shape == (1, 300, 4)
            data['encoder'] = cpu(out['interm_outputs']['pred_boxes'][0])
            data['initial'] = cpu(out['aux_outputs'][0]['pred_boxes'][0])
            data['final300'] = cpu(out['pred_boxes'][0])
            data['class_scores'] = cpu(out['pred_logits'][0].sigmoid())
            data['top300_ids'] = cpu(loc['topk_proposals'][0])
        def model_return(loc, result):
            if result is not None:
                data['padded_hw'] = tuple(loc['images'].tensor.shape[-2:])
                data['mapped_hw'] = tuple(loc['images'].image_sizes[0])
        def instance_return(loc, result):
            if result is not None:
                data['top100_ids'] = cpu(loc['topk_indices'])
        handlers = {attachment.compiled.__func__.__code__: decoder_return,
                    model.forward.__func__.__code__: model_return,
                    model.instance_inference.__func__.__code__: instance_return} if export else {}
        try:
            with torch.no_grad(), ReturnObserver(handlers):
                instances = model([mapped])[0]['instances']
        finally:
            handlers.clear()  # no callback/self cycles retaining large outputs
        if export:
            data['mask_native'] = cpu(attachment.last['native'].sigmoid()[0])
            arrays = {name+'_boxes_xyxy': to_original(data[name], data['padded_hw'], data['mapped_hw'],
                (image['height'], image['width'])) for name in ('encoder', 'mask_native', 'initial', 'final300')}
            arrays.update(top300_ids=data['top300_ids'], top100_ids=data['top100_ids'],
                class_scores=data['class_scores'], top100_boxes_xyxy=cpu(instances.pred_boxes.tensor),
                top100_scores=cpu(instances.scores))
            np.testing.assert_allclose(arrays['final300_boxes_xyxy'][data['top100_ids']],
                                       arrays['top100_boxes_xyxy'], rtol=1e-6, atol=1e-3)
            np.savez_compressed(diagnostics/f"{image['id']}.npz", **arrays)
            attachment.last = {}
        if instances.has('pred_masks'):
            instances.remove('pred_masks')
        rows = instances_to_coco_json(instances.to('cpu'), image['id'])
        for row in rows:
            assert row['category_id'] == 0
            row['category_id'] = 1
        return rows

    source = experiment['dataset']['val']
    first = read_json(source['annotations'])['images'][0]
    actual = predict(first, source['image_root'])
    expected = [r for r in read_json(job['source_predictions']) if r['image_id'] == first['id']]
    assert len(actual) == len(expected)
    np.testing.assert_allclose([r['bbox'] for r in actual], [r['bbox'] for r in expected], rtol=1e-5, atol=1e-3)
    np.testing.assert_allclose([r['score'] for r in actual], [r['score'] for r in expected], rtol=1e-5, atol=1e-6)
    write_json(output/'source_prediction_parity.json', {'status': 'passed', 'strict_checkpoint_load': True,
        'source_image_id': first['id'], 'bbox_atol': .001, 'score_atol': .000001})
    rows = []
    images = read_json(job['all_images'])['images']
    start = time.monotonic()
    for i, image in enumerate(images, 1):
        rows.extend(predict(image, job['image_root'], export=True))
        if i % 50 == 0 or i == len(images):
            write_json(output/'progress.json', {'images_done': i, 'images_total': len(images),
                'elapsed_seconds': time.monotonic()-start})
    assert versions == {k: p._version for k, p in model.state_dict(keep_vars=True).items()}
    write_json(output/'predictions.all_stills.bbox.json', rows)
    write_json(output/'verification.json', {'status': 'passed', 'images': len(images),
        'image_only_forward': True, 'native_tensors_unchanged': True,
        'checkpoint_sha256': sha256(job['checkpoint']), 'elapsed_seconds': time.monotonic()-start})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    run(parser.parse_args().job)
