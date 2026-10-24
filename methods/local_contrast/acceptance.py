"""Real source-data/GPU engineering gate. No AP, no target data, no updates."""
import argparse
import copy
import hashlib
from dataclasses import asdict, replace
from pathlib import Path
import sys
import random
import numpy as np
import torch
from polypbench.io import read_json, write_json
from .config import Options
from .monitor import native_digest, optimizer_contract


def seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run(output):
    from methods.mssl.config import build_cfg, MSSLOptions
    root = Path.cwd().resolve()
    exp = read_json(root/'runs/mssl-official50-s42-20261018T025946/experiment.json')
    exp['mssl'] = asdict(MSSLOptions(**exp['mssl']))
    exp['local_contrast'] = asdict(Options())
    sys.path.insert(0, exp['repo']['path'])
    import train_net
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.data import DatasetMapper
    from methods.mssl.data import load_records as base_records, MSSLMapper
    from .data import load_records, LocalMapper
    output.mkdir(parents=True, exist_ok=False)
    seed()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    cfg = build_cfg(exp, output)
    state_cpu = torch.get_rng_state().clone()
    model = build_model(cfg)
    state_constructed = torch.get_rng_state().clone()
    no_extra = copy.deepcopy(exp)
    no_extra.pop('local_contrast')
    torch.set_rng_state(state_cpu)
    native = build_model(build_cfg(no_extra, output))
    assert torch.equal(state_constructed, torch.get_rng_state()), 'Construction shifts native RNG'
    assert native_digest(model) == native_digest(native), 'Native initialization differs'
    del native
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    initialized_digest = native_digest(model)
    optimizer_contract(model, train_net.Trainer.build_optimizer(cfg, model))
    train = exp['dataset']['train']
    records = load_records(train['annotations'], train['image_root'], 'local_accept_train', exp['dataset']['raw_mask_root'])
    original = base_records(train['annotations'], train['image_root'], 'local_accept_base', exp['dataset']['raw_mask_root'])
    by_id = {r['image_id']: r for r in original}
    single = [r for r in records if r['mssl_source_single']]
    multi = [r for r in records if not r['mssl_source_single']]
    chosen = single[:6]+sorted(single, key=lambda r: r['annotations'][0]['bbox'][2]*r['annotations'][0]['bbox'][3])[:4]+multi[:4]
    mapper, base_mapper = LocalMapper(cfg, 'annotations'), MSSLMapper(cfg, 'annotations')
    results = []
    gradient_passes = 0
    captured = {}
    def capture(module, arguments, result):
        digest = hashlib.sha256()
        def visit(value):
            if isinstance(value, torch.Tensor):
                digest.update(str((value.shape, value.dtype)).encode())
                digest.update(value.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
            elif isinstance(value, dict):
                for key in sorted(value):
                    digest.update(str(key).encode())
                    visit(value[key])
            elif isinstance(value, (tuple, list)):
                for item in value:
                    visit(item)
        visit(result)
        captured['sha256'] = digest.hexdigest()
    handle = model.sem_seg_head.predictor.register_forward_hook(capture)
    for i, record in enumerate(chosen):
        seed(420+i)
        mapped = mapper(record)
        seed(420+i)
        expected = base_mapper(by_id[record['image_id']])
        for key in ('image', 'padding_mask'):
            assert torch.equal(mapped[key], expected[key]), 'Mapper parity '+key
        for key in ('gt_masks', 'gt_classes', 'gt_mssl_eligible'):
            assert torch.equal(mapped['instances'].get(key), expected['instances'].get(key)), 'Label parity '+key
        assert torch.equal(mapped['instances'].gt_boxes.tensor, expected['instances'].gt_boxes.tensor)
        model.train()
        model.local_options = Options(variant='off')
        seed(700+i)
        with torch.autocast('cuda', dtype=torch.float16):
            losses0 = model([expected])
        losses0 = {k: v.detach() for k, v in losses0.items()}
        outputs0 = captured['sha256']
        rng_cpu, rng_cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        receipt = dict(image_id=record['image_id'], original_single=record['mssl_source_single'], variants={})
        seed(700+i)
        with torch.autocast('cuda', dtype=torch.float16):
            repeat = model([expected])
        assert captured['sha256'] == outputs0, 'Native output self-repeat changed'
        receipt['native_repeat_loss_max_abs'] = max(float((repeat[k].detach()-losses0[k]).abs()) for k in losses0)
        del repeat
        for variant in ('box', 'mask_priority'):
            model.local_options = Options(variant=variant)
            model.local_sequence = 0
            model.local_probe_active = True
            seed(700+i)
            with torch.autocast('cuda', dtype=torch.float16):
                losses = model([mapped])
            assert set(losses)-set(losses0) == {'loss_local_contrast'}
            assert captured['sha256'] == outputs0, 'Native prediction tensors changed'
            assert torch.equal(rng_cpu, torch.get_rng_state()) and torch.equal(rng_cuda, torch.cuda.get_rng_state())
            differences = {key: float((losses[key].detach()-losses0[key]).abs()) for key in losses0}
            print('loss differences', {k: v for k, v in differences.items() if v},
                  'native repeat max', receipt['native_repeat_loss_max_abs'], flush=True)
            for key in losses0:
                torch.testing.assert_close(losses[key].detach(), losses0[key],
                    rtol=2*torch.finfo(torch.float32).eps, atol=0, msg='Native loss parity '+key)
            probes = model.last_local_probe
            if model.last_local_stats['used']:
                assert probes['gradients']['local']['finite'] and probes['gradients']['local']['feature_l2'] > 0
                param = next(p for n, p in model.backbone.named_parameters() if 'res3.0.conv1.weight' in n)
                grad = torch.autograd.grad(losses['loss_local_contrast']*64, param, allow_unused=True)[0]
                assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, 'No upstream backbone gradient'
                gradient_passes += 1
            receipt['variants'][variant] = copy.deepcopy(probes)
            receipt['variants'][variant]['native_loss_max_abs'] = max(differences.values())
            del losses
        a, b = receipt['variants']['box']['stats'], receipt['variants']['mask_priority']['stats']
        for key in ('total', 'used', 'skipped', 'fg_points', 'bg_points', 'budget_shortfall', 'box_fallback'):
            assert a[key] == b[key], 'Paired treatment differs: '+key
        results.append(receipt)
        write_json(output/'cases.json', results)
        print(f'acceptance {i+1}/{len(chosen)} image={record["image_id"]} used={a["used"]}', flush=True)
    assert gradient_passes > 0
    handle.remove()
    model.local_probe_active = False
    model.eval()
    item = DatasetMapper(cfg, False)(copy.deepcopy(records[0]))
    with torch.no_grad():
        model.local_options = Options(variant='off')
        a = model([item])[0]['instances']
        model.local_options = Options(variant='mask_priority')
        b = model([item])[0]['instances']
    for key in ('pred_boxes', 'scores', 'pred_masks'):
        x, y = a.get(key), b.get(key)
        if key == 'pred_boxes':
            x, y = x.tensor, y.tensor
        assert torch.equal(x, y), 'Image-only inference changes with loss switch'
    assert native_digest(model) == initialized_digest, 'Read-only gate modified weights/buffers'
    write_json(output/'verification.json', dict(status='passed', cases=len(results),
        native_prediction_tensors_rng_exact=True, native_scalar_loss_rtol=2*torch.finfo(torch.float32).eps,
        native_scalar_loss_atol=0, mapper_native_labels_exact=True, construction_rng_exact=True,
        shared_stride8_feature_and_upstream_gradients=True, no_new_parameters=True,
        initialization_sha256=initialized_digest, image_only_inference_exact=True,
        optimizer_updates=0, ap_evaluation=False, target_data_used=False,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args().output.resolve())
