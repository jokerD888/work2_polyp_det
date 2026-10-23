"""Finite source-only real-detector identity, routing and gradient acceptance."""
import argparse
import copy
from dataclasses import asdict
import gc
from pathlib import Path
import sys
import time
import traceback
import torch
from polypbench.io import read_json, write_json, sha256
from polypbench.analysis.support_reference_acceptance import Capture, tensor_digest, panel_from_annotations, seed_all, timing
from polypbench.analysis.candidate_stage import ReturnObserver
from .integration import ReferenceFeaturesAttachment


def compare_batch_summary(expected, actual):
    # Raw predictions/references remain bitwise. Native CUDA reductions may
    # differ by float32 rounding even with identical mask/logit tensors.
    assert {k: v for k, v in expected.items() if k != 'losses'} == {
        k: v for k, v in actual.items() if k != 'losses'}, 'Native prediction/reference changed'
    assert set(expected['losses']) == set(actual['losses'])
    differences = {}
    for key, value in expected['losses'].items():
        a, b = torch.tensor(value), torch.tensor(actual['losses'][key])
        torch.testing.assert_close(a, b, rtol=2*torch.finfo(torch.float32).eps, atol=0)
        if value != actual['losses'][key]:
            differences[key] = actual['losses'][key]-value
    return differences


def capture(model, data, attachment=None, graph=False, amp=True):
    seed_all(2718)
    code = (attachment.compiled.__func__.__code__ if attachment and attachment.enabled
        else attachment.original.__func__.__code__ if attachment else model.sem_seg_head.predictor.forward.__func__.__code__)
    observer = Capture(model, code)
    try:
        with torch.set_grad_enabled(graph), torch.autocast('cuda', enabled=amp), ReturnObserver(observer.handlers):
            result = model(data)
    finally:
        observer.handlers.clear()
    if model.training:
        observer.summary['losses'] = {k: float(v.detach()) for k, v in result.items()}
        assert all(torch.isfinite(v) for v in result.values())
    if attachment and attachment.enabled:
        torch.testing.assert_close(observer.out['aux_outputs'][0]['pred_boxes'], attachment.last['native'].sigmoid(), rtol=0, atol=0)
        assert attachment.last['dn_cross_unchanged']
        torch.testing.assert_close(attachment.last['extra_reference'],
            attachment.last['main_reference'] if attachment.variant == 'mask_extra' else
            attachment.last['encoder'].sigmoid()[:, :, None]*torch.cat(
                [attachment.last['valid_ratios'], attachment.last['valid_ratios']], -1)[:, None], rtol=0, atol=0)
    return result, observer


def gradients(losses, adapter):
    params = list(adapter.parameters())
    groups = {'initial': ('loss_bbox_0', 'loss_giou_0'),
        'first_decoder': ('loss_bbox_1', 'loss_giou_1'),
        'encoder': ('loss_bbox_interm', 'loss_giou_interm'),
        'first_decoder_dn': ('loss_bbox_dn_1', 'loss_giou_dn_1')}
    rows = {}
    for label, keys in groups.items():
        selected = [losses[k] for k in keys if k in losses]
        if not selected:
            continue
        gs = torch.autograd.grad(sum(selected), params, retain_graph=True, allow_unused=True)
        assert all(g is None or torch.isfinite(g).all() for g in gs)
        rows[label] = {name: float(g.float().norm()) if g is not None else 0.
            for (name, _), g in zip(adapter.named_parameters(), gs)}
    assert all(v == 0 for label in ('initial', 'encoder', 'first_decoder_dn') for v in rows.get(label, {}).values())
    return rows


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    root = Path.cwd()
    start = time.monotonic()
    write_json(output/'state.json', {'status': 'running'})
    try:
        sys.path.insert(0, str(root/'third_party/MaskDINO'))
        import train_net
        from detectron2.modeling import build_model
        from detectron2.data import DatasetMapper
        from methods.mssl.config import build_cfg, MSSLOptions
        from methods.mssl.data import load_records, MSSLMapper
        job = read_json(root/'runs/initialization-intervention-full-20261021-r2/kvasir_val_baseline.job.json')
        experiment = read_json(job['experiment'])
        experiment['mssl'] = asdict(MSSLOptions(**experiment['mssl']).validate())
        cfg = build_cfg(experiment, output, device='cuda')
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = False
        seed_all(42)
        model = build_model(cfg)
        saved = torch.load(job['checkpoint'], map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'], strict=True)
        del saved
        train = experiment['dataset']['train']
        panel = panel_from_annotations(read_json(train['annotations']))
        # Deterministic source strata: two scale quartiles, single/multiple boxes.
        images = panel['images'][::4]
        write_json(output/'panel.json', {'images': images, 'source': 'Kvasir Train', 'chosen_without_predictions': True})
        records = load_records(train['annotations'], train['image_root'],
            'reference_features_acceptance_train', experiment['dataset']['raw_mask_root'])
        by_id = {r['image_id']: r for r in records}
        mapper, eval_mapper = MSSLMapper(cfg, 'annotations'), DatasetMapper(cfg, False)
        rows = {'eval': [], 'train': [], 'batch2': [], 'nonzero': [], 'timing': {}}
        native_versions = {n: p._version for n, p in model.state_dict(keep_vars=True).items()}
        model.eval()
        expected_eval = {}
        for image in images:
            data = [eval_mapper(copy.deepcopy(by_id[image['id']]))]
            _, observed = capture(model, data, amp=False)
            expected_eval[image['id']] = observed.summary
            del observed
        for variant in ('mask_extra', 'encoder_extra'):
            cpu_rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
            with ReferenceFeaturesAttachment(model, variant, capture=True) as attachment:
                assert torch.equal(cpu_rng, torch.random.get_rng_state())
                assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
                model.eval()
                for image in images:
                    data = [eval_mapper(copy.deepcopy(by_id[image['id']]))]
                    for enabled in (False, True):
                        attachment.enabled = enabled
                        _, observed = capture(model, data, attachment, amp=False)
                        assert observed.summary == expected_eval[image['id']], 'FP32 native identity failed'
                        del observed
                    rows['eval'].append({'variant': variant, 'image': image['id'], 'disabled_and_zero_bitwise': True})
                model.train()
                for image in images:
                    for lsj_seed in (42, 73):
                        seed_all(lsj_seed)
                        data = [mapper(copy.deepcopy(by_id[image['id']]))]
                        attachment.enabled = False
                        losses, before = capture(model, data, attachment, graph=True)
                        expected = before.summary
                        del losses, before
                        attachment.enabled = True
                        losses, observed = capture(model, data, attachment, graph=True)
                        assert observed.summary == expected, 'AMP native identity failed'
                        gs = gradients(losses, attachment.adapter)
                        if len(data[0]['instances']):
                            assert gs['first_decoder']['projection.weight'] > 0, 'No matched first-decoder signal'
                        rows['train'].append({'variant': variant, 'image': image['id'], 'lsj_seed': lsj_seed,
                            'gt': len(data[0]['instances']), 'dn_pad': attachment.last['dn_pad'], 'gradients': gs, 'native_identity': True})
                        del losses, observed, data
                        attachment.last = {}
                        gc.collect()
                        write_json(output/'results.json', rows)
                seed_all(42)
                data = [mapper(copy.deepcopy(by_id[image['id']])) for image in images[:2]]
                attachment.enabled = False
                _, a = capture(model, data, attachment)
                expected = a.summary
                del a
                _, repeated = capture(model, data, attachment)
                native_rounding = compare_batch_summary(expected, repeated.summary)
                del repeated
                attachment.enabled = True
                _, b = capture(model, data, attachment)
                rounding = compare_batch_summary(expected, b.summary)
                rows['batch2'].append({'variant': variant, 'native_predictions_bitwise': True,
                    'native_loss_repeat_rounding': native_rounding, 'extra_loss_rounding': rounding,
                    'loss_rtol': 2*torch.finfo(torch.float32).eps, 'loss_atol': 0,
                    'dn_pad': attachment.last['dn_pad']})
                del b, data
                # Fixed nonzero wiring probe. Neither fitted nor used in formal initialization.
                with torch.no_grad():
                    attachment.adapter.projection.weight.copy_(.001*torch.eye(256, device='cuda'))
                seed_all(42)
                data = [mapper(copy.deepcopy(by_id[images[0]['id']]))]
                losses, probe = capture(model, data, attachment, graph=True)
                assert attachment.last['residual_l2'] > 0
                gs = gradients(losses, attachment.adapter)
                assert any(value > 0 for name, value in gs['first_decoder'].items() if name.startswith('sampler.'))
                rows['nonzero'].append({'variant': variant, 'gradients': gs, 'dn_prefix_unchanged': True,
                    'native_initial_reference_unchanged': True})
                del losses, probe, data
                with torch.no_grad():
                    attachment.adapter.projection.weight.zero_()
                model.eval()
                data = [eval_mapper(copy.deepcopy(by_id[images[0]['id']]))]
                rows['timing'][variant] = timing(model, data, attachment)
                assert native_versions == {n: p._version for n, p in model.state_dict(keep_vars=True).items() if '.reference_features.' not in n}
                attachment.last = {}
            gc.collect()
            torch.cuda.empty_cache()
        write_json(output/'results.json', rows)
        write_json(output/'verification.json', {'status': 'passed', 'source_train_only': True,
            'optimizer_steps': 0, 'no_ap_evaluation': True, 'eval_cases': len(rows['eval']),
            'train_cases': len(rows['train']), 'batch2_cases': len(rows['batch2']),
            'native_tensors_unchanged': True, 'native_initial_reference_unchanged': True,
            'extra_direct_dn_residual': False, 'elapsed_seconds': time.monotonic()-start})
        write_json(output/'state.json', {'status': 'completed'})
    except BaseException as error:
        write_json(output/'state.json', {'status': 'failed', 'error': repr(error), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args().output.resolve())
