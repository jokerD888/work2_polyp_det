"""Finite real-Detector engineering acceptance, without optimizer or training.

Uses Kvasir Train only. The panel is fixed from annotations before loading any
model. Frozen mature weights test routing, NOT learnability/convergence.
"""
import argparse
from collections import defaultdict
import copy
import gc
import hashlib
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from polypbench.io import read_json, write_json, sha256
from .candidate_stage import ReturnObserver, NATIVE_COMMIT, remap_experiment
from methods.support_reference.integration import SupportReferenceAttachment, load_native_checkpoint


SEEDS = (42, 31415)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_digest(tensor):
    if tensor is None:  # Native eval intentionally omits auxiliary masks.
        return None
    array = tensor.detach().contiguous().cpu().view(torch.uint8).numpy()
    return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
            'sha256': hashlib.sha256(array.tobytes()).hexdigest()}


def assert_summary_equal(expected, actual, output, label):
    if expected != actual:
        write_json(output/f'{label}.parity_failure.json', {'expected': expected, 'actual': actual,
            'different_top_level_keys': [k for k in expected if expected[k] != actual.get(k)]})
        raise AssertionError(f'{label}: strict parity failed; details saved')


def panel_from_annotations(gt):
    annotations = defaultdict(list)
    for ann in gt['annotations']:
        if not ann.get('iscrowd', 0):
            annotations[ann['image_id']].append(ann)
    rows = []
    for image in gt['images']:
        anns = annotations[image['id']]
        if not anns:
            continue
        rows.append({**image, 'gt': len(anns), 'minimum_relative_box_area': min(
            a['bbox'][2] * a['bbox'][3] / (image['height'] * image['width']) for a in anns)})
    cuts = np.quantile([r['minimum_relative_box_area'] for r in rows], [.25, .5, .75])
    groups = defaultdict(list)
    for row in rows:
        row['stratum'] = ('single' if row['gt'] == 1 else 'multi') + '/q' + str(
            1 + int(np.searchsorted(cuts, row['minimum_relative_box_area'], side='left')))
        groups[row['stratum']].append(row)
    selected = []
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda r: hashlib.sha256(
            f"support-reference-panel-v0/42/{r['id']}".encode()).hexdigest())
        selected.extend(ordered[:2])
    return {'selection': 'two annotation-only hash-ranked images per multiplicity/minimum-area quartile',
        'minimum_relative_box_area_quartile_cuts': cuts.tolist(),
        'stratum_population': {k: len(v) for k, v in groups.items()},
        'images': selected, 'lsj_seeds': list(SEEDS), 'population_prevalence_estimate': False}


class Capture:
    def __init__(self, model, decoder_code):
        self.model, self.decoder_code = model, decoder_code
        self.out, self.mask_dict = None, None
        self.matches = []
        self.summary = {}
        self.handlers = {decoder_code: self.decoder_return,
            model.forward.__func__.__code__: self.model_return,
            model.instance_inference.__func__.__code__: self.instance_return,
            model.criterion.matcher.memory_efficient_forward.__wrapped__.__code__: self.matcher_return}

    def decoder_return(self, loc, result):
        if result is None:
            return
        self.out, self.mask_dict = result
        out = self.out
        summary = {'top_ids': tensor_digest(loc['topk_proposals']),
            'ordinary_content': tensor_digest(loc['tgt'][:, -300:]),
            'encoder_reference': tensor_digest(loc['refpoint_embed_undetach']),
            'actual_reference': tensor_digest(loc['refpoint_embed']),
            'early_mask': tensor_digest(out['interm_outputs']['pred_masks']),
            'final_mask': tensor_digest(out['pred_masks']),
            'final_class': tensor_digest(out['pred_logits']),
            'final_boxes': tensor_digest(out['pred_boxes']),
            'initial_mask': tensor_digest(out['aux_outputs'][0]['pred_masks']),
            'auxiliary': [{k: tensor_digest(a[k]) for k in ('pred_logits', 'pred_boxes')}
                          for a in out['aux_outputs']],
            'ordinary_queries': out['pred_boxes'].shape[1],
            'batch': out['pred_boxes'].shape[0]}
        if self.mask_dict is not None:
            known = self.mask_dict['output_known_lbs_bboxes']
            self.dn_initial = loc['input_query_bbox'].detach()
            actual = loc['refpoint_embed']
            dn = actual[:, :self.mask_dict['pad_size']]
            torch.testing.assert_close(dn, self.dn_initial, rtol=0, atol=0)
            torch.testing.assert_close(known['aux_outputs'][0]['pred_boxes'], dn.sigmoid(), rtol=0, atol=0)
            summary.update(dn_input=tensor_digest(dn), dn_pad=self.mask_dict['pad_size'],
                dn_attention=tensor_digest(loc['tgt_mask']),
                dn_initial_boxes=tensor_digest(known['aux_outputs'][0]['pred_boxes']),
                dn_final_boxes=tensor_digest(known['pred_boxes']))
        else:
            summary['dn_pad'] = 0
        assert len(out['aux_outputs']) == 9 and out['pred_boxes'].shape[1] == 300
        self.summary.update(summary)

    def matcher_return(self, loc, result):
        if result is not None and loc.get('self') is self.model.criterion.matcher:
            self.matches.append((id(loc['outputs']),
                [(a.tolist(), b.tolist()) for a, b in result], loc['targets']))

    def model_return(self, loc, result):
        if result is not None and loc.get('self') is self.model:
            self.summary['padded_hw'] = list(loc['images'].tensor.shape[-2:])
            self.summary['mapped_hw'] = [list(s) for s in loc['images'].image_sizes]

    def instance_return(self, loc, result):
        if result is not None:
            self.summary['instance'] = {k: tensor_digest(v) for k, v in {
                'boxes': result.pred_boxes.tensor, 'scores': result.scores,
                'classification': loc['scores_per_image'], 'mask_quality': loc['mask_scores_per_image'],
                'query_ids': loc['topk_indices']}.items()}

    def aux0_assignment(self):
        matches = [m for m in self.matches if m[0] == id(self.out['aux_outputs'][0])]
        assert len(matches) == 1, 'aux0 must use exactly its native matcher call'
        return matches[0][1:]


def forward_capture(model, data, seed, attach=None, graph=False, amp=True):
    seed_all(seed)
    code = (attach.compiled.__func__.__code__ if attach and attach.enabled
            else (attach.original.__func__.__code__ if attach else model.sem_seg_head.predictor.forward.__func__.__code__))
    capture = Capture(model, code)
    try:
        with torch.set_grad_enabled(graph), torch.autocast('cuda', enabled=amp), ReturnObserver(capture.handlers):
            result = model(data)
    finally:
        # Bound callbacks otherwise create Capture -> handlers -> Capture cycles,
        # delaying release of the entire graph until cyclic GC and doubling VRAM.
        capture.handlers.clear()
    if model.training:
        capture.summary['losses'] = {k: float(v.detach()) for k, v in result.items()}
        assert all(torch.isfinite(v) for v in result.values()), 'nonfinite native loss'
    if attach and attach.enabled:
        last = attach.last
        assert list(last['canvas_hw']) == capture.summary['padded_hw']
        torch.testing.assert_close(capture.out['aux_outputs'][0]['pred_boxes'],
                                    last['actual'].sigmoid(), rtol=0, atol=0)
    return result, capture


def gradient_groups(model, attachment, losses):
    params = list(attachment.adapter.parameters())
    native = [model.sem_seg_head.predictor._bbox_embed.layers[-1].bias,
              model.sem_seg_head.predictor.mask_embed.layers[-1].bias]
    groups = {
        'aux0_box': ['loss_bbox_0', 'loss_giou_0'],
        'first_decoder_box': ['loss_bbox_1', 'loss_giou_1'],
        'final_box': ['loss_bbox', 'loss_giou'],
        'final_class_mask': ['loss_ce', 'loss_mask', 'loss_dice'],
        'encoder_intermediate': [k for k in losses if k.endswith('_interm')],
        'dn_initial': [k for k in losses if k.endswith('_dn_0')],
        'dn_later': [k for k in losses if '_dn' in k and not k.endswith('_dn_0')],
        'mssl_final_only': ['loss_mssl'] if 'loss_mssl' in losses else [],
        'total': list(losses)}
    rows = {}
    for label, keys in groups.items():
        keys = [k for k in keys if k in losses]
        if not keys:
            continue
        loss = sum(losses[k] for k in keys)
        scale = 64.
        gradients = (torch.autograd.grad(loss * scale, params + native,
            retain_graph=True, allow_unused=True) if loss.requires_grad else [None] * (len(params) + len(native)))
        finite = all(g is None or bool(torch.isfinite(g).all()) for g in gradients)
        assert finite, f'nonfinite {label} gradient'
        def norm(values):
            terms = [(g.detach().float() / scale).square().sum() for g in values if g is not None]
            return float(torch.stack(terms).sum().sqrt()) if terms else 0.
        rows[label] = {'weighted_loss': float(loss.detach()), 'keys': keys,
            'adapter_l2': norm(gradients[:len(params)]),
            'adapter_first_layer_l2': norm(gradients[:2]),
            'adapter_last_layer_l2': norm(gradients[2:len(params)]),
            'native_bbox_last_bias_l2': norm(gradients[-2:-1]),
            'native_mask_last_bias_l2': norm(gradients[-1:]),
            'adapter_used': sum(g is not None for g in gradients[:len(params)]),
            'finite': finite, 'measurement_scale_divided_out': scale}
    assert rows['encoder_intermediate']['adapter_l2'] == 0
    if 'dn_initial' in rows:
        assert rows['dn_initial']['adapter_l2'] == 0
    assert rows['aux0_box']['native_bbox_last_bias_l2'] == 0
    assert rows['aux0_box']['native_mask_last_bias_l2'] == 0
    assert all(p.grad is None for p in model.parameters()), 'autograd.grad must not write .grad'
    return rows


def supervision_record(capture, attachment, data):
    from maskdino.utils import box_ops
    matches, targets = capture.aux0_assignment()
    last = attachment.last
    rows = []
    for i, (sources, destinations) in enumerate(matches):
        for source, destination in zip(sources, destinations):
            a = last['native'][i, source].sigmoid().float()[None]
            b = targets[i]['boxes'][destination].float()[None]
            iou = box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(a), box_ops.box_cxcywh_to_xyxy(b))[0].item()
            gt_mask = targets[i]['masks'][destination]
            rows.append({'batch': i, 'query': source, 'gt': destination,
                'adapter_eligible': bool(last['eligible'][i, source]), 'initial_box_iou': iou,
                'gt_mask_pixels': int(gt_mask.sum()),
                'gt_box_relative_area': float(b[0, 2] * b[0, 3]),
                'mask_occupancy': float(last['features'][i, source, -8]),
                'padding_foreground_fraction': float(last['features'][i, source, -6]),
                'original_mssl_mask_eligible': bool(data[i]['instances'].gt_mssl_eligible[destination])})
    return {'all_query_eligible': int(last['eligible'].sum()),
        'all_queries': last['eligible'].numel(), 'matches': rows,
        'post_lsj_gt': sum(len(d['instances']) for d in data),
        'all_retained_gt_masks_empty': sum(int((d['instances'].gt_masks.flatten(1).sum(1) == 0).sum()) for d in data),
        'valid_canvas_hw': list(last['canvas_hw']),
        'valid_lattice_cells': [int(x.sum()) for x in last['valid_lattice']]}


def timing(model, data, attachment):
    attachment.capture = False
    times, memory = {}, {}
    model.eval()
    for enabled in (False, True):
        attachment.enabled = enabled
        with torch.no_grad():
            model(data)  # warm up current shape and branch
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            durations = []
            for _ in range(5):
                start = time.perf_counter()
                model(data)
                torch.cuda.synchronize()
                durations.append(time.perf_counter() - start)
        times[str(enabled)] = durations
        memory[str(enabled)] = torch.cuda.max_memory_allocated()
    attachment.capture = True
    return {'seconds': times, 'peak_allocated_bytes': memory,
            'scope': 'five warmed image-only forwards, no observer, batch1, FP32, engineering panel only'}


def _run(output):
    root = Path.cwd()
    start = time.monotonic()
    sys.path.insert(0, str(root / 'third_party/MaskDINO'))
    import train_net  # noqa: F401
    from detectron2.modeling import build_model
    from detectron2.data import DatasetMapper
    from methods.mssl.config import build_cfg
    from methods.mssl.data import load_records, MSSLMapper
    assert subprocess.check_output(['git', '-C', str(root/'third_party/MaskDINO'), 'rev-parse', 'HEAD'], text=True).strip() == NATIVE_COMMIT
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    jobs = {name: read_json(root / f'runs/initialization-intervention-full-20261021-r2/kvasir_val_{name}.job.json')
            for name in ('baseline', 'mssl')}
    experiments = {k: remap_experiment(read_json(v['experiment']), root) for k, v in jobs.items()}
    exp = experiments['baseline']
    train = exp['dataset']['train']
    panel = panel_from_annotations(read_json(train['annotations']))
    write_json(output/'panel.json', panel)
    assert 0 < len(panel['images']) <= 16
    paths = [train['annotations'], jobs['baseline']['checkpoint'], jobs['mssl']['checkpoint'],
             jobs['baseline']['experiment'], jobs['mssl']['experiment']]
    for image in panel['images']:
        paths.extend((str(Path(train['image_root']) / image['file_name']),
                      str(Path(exp['dataset']['raw_mask_root']) / Path(image['file_name']).name)))
    source_paths = [str(p) for p in (root/'methods/support_reference').glob('*.py')]
    source_paths += [str(Path(__file__)), str(root/'polypbench/analysis/support_reference_prototype.py')]
    fingerprints = {p: sha256(p) for p in paths + source_paths}
    write_json(output/'protocol.json', {'training': False, 'optimizer_steps': 0,
        'cohort': 'Kvasir Train only', 'checkpoint_scope': 'mature Baseline and MSSL weights; no convergence claim',
        'baseline_train_cases': len(panel['images']) * len(SEEDS),
        'mssl_train_cases': len(panel['images'][::4]) * len(SEEDS),
        'train_precision': 'native-resolution1024 AMP FP16', 'eval_precision': 'FP32',
        'source_sha256': fingerprints, 'panel': 'panel.json',
        'nonzero_probe': 'fixed last-layer bias [0.01,-0.01,0.005,-0.005], not optimized',
        'dn_semantics': 'DN references unchanged; later DN may indirectly attend ordinary queries',
        'valid_cell_policy': 'nearest-exact cell-center representative of LSJ and batch valid canvas'})
    source_dir = output/'source'
    for path in source_paths:
        relative = Path(path).relative_to(root)
        target = source_dir/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(path).read_bytes())
    results = {'eval': [], 'train': [], 'nonzero': [], 'batch2': [], 'timing': {}}
    records = load_records(train['annotations'], train['image_root'],
                           'support_reference_acceptance_train', exp['dataset']['raw_mask_root'])
    by_id = {r['image_id']: r for r in records}
    try:
        for name in ('baseline', 'mssl'):
            seed_all(42)
            cfg = build_cfg(experiments[name], output, device='cuda')
            model = build_model(cfg)
            checkpoint = torch.load(jobs[name]['checkpoint'], map_location='cpu', weights_only=False)['model']
            model.load_state_dict(checkpoint, strict=True)
            model.eval()
            versions = {k: v._version for k, v in model.state_dict(keep_vars=True).items()}
            mapper = MSSLMapper(cfg, 'annotations')
            eval_mapper = DatasetMapper(cfg, False)
            # First image per fixed stratum, then one single + one multi per two quartiles.
            eval_images = panel['images'][::4]
            native_eval = {}
            for image in eval_images:
                mapped = eval_mapper(copy.deepcopy(by_id[image['id']]))
                _, capture = forward_capture(model, [mapped], 101, amp=False)
                native_eval[image['id']] = capture.summary
                del capture
            model.train()
            first = panel['images'][0]
            seed_all(SEEDS[0])
            native_train_data = [mapper(copy.deepcopy(by_id[first['id']]))]
            reference_losses, reference = forward_capture(model, native_train_data, 2718, graph=True)
            native_train_reference = reference.summary
            del reference_losses, reference, native_train_data
            model.eval()
            cpu_rng = torch.random.get_rng_state().clone()
            cuda_rng = torch.cuda.get_rng_state().clone()
            with SupportReferenceAttachment(model, enabled=False, capture=True) as attachment:
                assert torch.equal(cpu_rng, torch.random.get_rng_state())
                assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
                missing = load_native_checkpoint(model, checkpoint)
                # Reload was explicit; reset the read-only tensor-version baseline.
                versions = {k: v._version for k, v in model.state_dict(keep_vars=True).items()
                            if '.support_reference.' not in k}
                del checkpoint
                for image in eval_images:
                    mapped = eval_mapper(copy.deepcopy(by_id[image['id']]))
                    row = {'model': name, 'image_id': image['id'], 'missing_only_adapter': missing}
                    for enabled in (False, True):
                        attachment.enabled = enabled
                        _, capture = forward_capture(model, [mapped], 101, attachment, amp=False)
                        assert capture.summary == native_eval[image['id']], 'FP32 inference is not bitwise native'
                        row['enabled_' + str(enabled)] = 'all captured predictions bitwise native'
                        del capture
                    results['eval'].append(row)
                train_images = panel['images'] if name == 'baseline' else panel['images'][::4]
                model.train()
                for image in train_images:
                    for lsj_seed in SEEDS:
                        seed_all(lsj_seed)
                        mapped = mapper(copy.deepcopy(by_id[image['id']]))
                        data = [mapped]
                        attachment.enabled = False
                        disabled_losses, disabled = forward_capture(model, data, 2718, attachment, graph=True)
                        expected = disabled.summary
                        if image['id'] == first['id'] and lsj_seed == SEEDS[0]:
                            assert_summary_equal(native_train_reference, expected, output, name+'_unattached_train')
                        del disabled_losses, disabled
                        attachment.enabled = True
                        losses, capture = forward_capture(model, data, 2718, attachment, graph=True)
                        assert_summary_equal(expected, capture.summary, output,
                            f'{name}_{image["id"]}_{lsj_seed}_zero_train')
                        gradients = gradient_groups(model, attachment, losses)
                        supervision = supervision_record(capture, attachment, data)
                        if any(r['adapter_eligible'] for r in supervision['matches']):
                            assert gradients['aux0_box']['adapter_l2'] > 0, 'no direct matched adapter supervision'
                        row = {'model': name, 'image_id': image['id'], 'stratum': image['stratum'],
                            'lsj_seed': lsj_seed, 'native_parity': True,
                            'dn_pad': capture.summary['dn_pad'], 'loss_keys': sorted(losses),
                            'supervision': supervision, 'gradients': gradients,
                            'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
                        results['train'].append(row)
                        del losses, capture, data
                        attachment.last = {}
                        gc.collect()
                        write_json(output/'results.json', results)
                        write_json(output/'state.json', {'status': 'running', 'phase': name,
                            'train_cases_done': len(results['train']), 'optimizer_steps': 0,
                            'elapsed_seconds': time.monotonic()-start})
                        print(f"{name} train {image['id']} seed{lsj_seed}: "
                              f"eligible {supervision['all_query_eligible']}/300, matches {len(supervision['matches'])}, "
                              f"aux0grad {gradients['aux0_box']['adapter_l2']:.5g}", flush=True)
                # A nonzero probe is a deterministic wiring test, never a fitted rule.
                image = train_images[0]
                seed_all(42)
                data = [mapper(copy.deepcopy(by_id[image['id']]))]
                with torch.no_grad():
                    attachment.adapter.head[-1].bias.copy_(torch.tensor([.01, -.01, .005, -.005], device='cuda'))
                losses, capture = forward_capture(model, data, 2718, attachment, graph=True)
                last = attachment.last
                assert not torch.equal(last['actual'][last['eligible']], last['native'][last['eligible']])
                assert torch.equal(last['actual'][~last['eligible']], last['native'][~last['eligible']])
                gradients = gradient_groups(model, attachment, losses)
                results['nonzero'].append({'model': name, 'dn_reference_unchanged': True,
                    'ordinary_only': True, 'gradients': gradients})
                del losses, capture, data
                attachment.last = {}
                with torch.no_grad():
                    attachment.adapter.head[-1].bias.zero_()
                # Actual two-image DN batching, unequal object counts where retained.
                seed_all(42)
                data = [mapper(copy.deepcopy(by_id[panel['images'][i]['id']])) for i in (0, 8)]
                attachment.enabled = False
                _, base = forward_capture(model, data, 2718, attachment)
                expected = base.summary
                del base
                attachment.enabled = True
                _, other = forward_capture(model, data, 2718, attachment)
                assert other.summary == expected
                results['batch2'].append({'model': name, 'batch': 2, 'native_parity': True,
                    'gt_counts': [len(d['instances']) for d in data], 'dn_pad': other.summary['dn_pad']})
                del other, data
                attachment.last = {}
                mapped = eval_mapper(copy.deepcopy(by_id[eval_images[0]['id']]))
                results['timing'][name] = timing(model, [mapped], attachment)
                current = {k: v._version for k, v in model.state_dict(keep_vars=True).items()
                           if '.support_reference.' not in k}
                assert current == versions, 'native parameter/buffer mutated'
            assert not any('.support_reference.' in k for k in model.state_dict())
            del model
            gc.collect()
            torch.cuda.empty_cache()
        assert all(sha256(p) == v for p, v in fingerprints.items()), 'source/input changed during acceptance'
        write_json(output/'results.json', results)
        write_json(output/'verification.json', {'status': 'passed', 'optimizer_steps': 0,
            'eval_cases': len(results['eval']), 'train_cases': len(results['train']),
            'native_tensors_unchanged': True, 'input_hashes_rechecked': len(fingerprints),
            'elapsed_seconds': time.monotonic()-start, 'torch': torch.__version__,
            'gpu': torch.cuda.get_device_name(), 'no_polypgen_inputs': True,
            'performance_or_novelty_claim': False})
        write_json(output/'state.json', {'status': 'completed', 'optimizer_steps': 0})
    except BaseException as error:
        write_json(output/'results.json', results)
        write_json(output/'state.json', {'status': 'failed', 'error': repr(error),
            'traceback': traceback.format_exc(), 'optimizer_steps': 0})
        raise


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'state.json', {'status': 'preparing', 'optimizer_steps': 0})
    try:
        return _run(output)
    except BaseException as error:
        write_json(output/'state.json', {'status': 'failed', 'error': repr(error),
            'traceback': traceback.format_exc(), 'optimizer_steps': 0})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(Path(args.output).resolve())


if __name__ == '__main__':
    main()
