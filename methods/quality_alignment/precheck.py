"""GPU routing/gradient acceptance on real source augmentations, no updates."""
import argparse
import copy
import hashlib
import random
import sys
from pathlib import Path
import numpy as np
import torch

from polypbench.io import read_json, write_json, sha256


def run(config, output, training=None):
    from methods.mssl.config import load_experiment, build_cfg
    e = load_experiment(config) if training is None else read_json(Path(training) / "experiment.json")
    sys.path.insert(0, e['repo']['path'])
    import train_net
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.data import DatasetMapper
    from detectron2.data.datasets import load_coco_json
    from maskdino.modeling.criterion import SetCriterion
    from maskdino.maskdino import MaskDINO
    from methods.mssl.model import MSSLCriterion
    from methods.mssl.data import load_records, MSSLMapper
    from methods.mssl.precheck import rng_state, restore
    from .model import QualityAlignMaskDINO
    from .probe import QualityGradientProbe

    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'checks': {}, 'panels': [], 'no_parameter_updates': True,
              'mssl_enabled': e['mssl']['enabled'], 'gpu': torch.cuda.get_device_name()}
    write_json(out / 'report.json', report)
    try:
        cfg = build_cfg(e, out)
        (out / 'resolved.yaml').write_text(cfg.dump(), encoding='utf-8')
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = False
        def fingerprint(module):
            digest = hashlib.sha256()
            for name, tensor in sorted(module.state_dict().items()):
                digest.update(name.encode()); digest.update(str(tensor.dtype).encode())
                digest.update(str(tuple(tensor.shape)).encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            return digest.hexdigest()
        if not training:
            without_q = copy.deepcopy(e); without_q.pop('quality_alignment')
            original = build_model(build_cfg(without_q, out))
            DetectionCheckpointer(original).load(cfg.MODEL.WEIGHTS)
            native_fingerprint = fingerprint(original)
            del original; torch.cuda.empty_cache()
            random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        model = build_model(cfg)
        if training:
            ck = Path(training) / 'train/model_best.pth'; selected = read_json(Path(training) / 'train/selected_checkpoint.json')
            assert sha256(ck) == selected['sha256']
            model.load_state_dict(torch.load(ck, map_location='cpu', weights_only=False)['model'], strict=True)
            report['checkpoint_sha256'] = selected['sha256']
        else:
            DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
            assert fingerprint(model) == native_fingerprint
            report['checks']['q_off_initialization_all_tensors_exact'] = native_fingerprint
        assert QualityAlignMaskDINO.forward is MaskDINO.forward
        assert not list(model.criterion.parameters())
        report['checks']['no_new_parameters_native_inference'] = True
        records = load_records(e['dataset']['train']['annotations'], e['dataset']['train']['image_root'],
                               'quality_accept_train', e['dataset']['raw_mask_root'])
        singles = sorted([r for r in records if r['mssl_source_single']], key=lambda r:r['image_id'])
        multis = sorted([r for r in records if not r['mssl_source_single']], key=lambda r:r['image_id'])
        candidates = [singles[0], multis[0]]
        mapper = MSSLMapper(cfg, 'annotations')
        model.train()
        for index, record in enumerate(candidates):
            seed = 42000 + index
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            batch = [mapper(record)]
            if not len(batch[0]['instances']):
                raise RuntimeError('Fixed panel has no instances')
            captured = {}; observed = []
            def capture(module, inputs):
                captured['inputs'] = inputs; captured['rng'] = rng_state()
            def observe(module, inputs, result):
                observed.append([(a.tolist(), b.tolist()) for a,b in result])
            hook = model.criterion.register_forward_pre_hook(capture)
            mh = model.criterion.matcher.register_forward_hook(observe)
            probe = QualityGradientProbe(model, out / 'weighted_class_gradients.jsonl', 1)
            model.criterion.probe_callback = probe.callback(index, 0)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                losses = model(batch)
            hook.remove(); model.criterion.probe_callback = None
            after = rng_state(); first_matches = list(observed); observed.clear()
            assert model.criterion._q_calls == 1
            assert all(torch.isfinite(v).all() for v in losses.values())
            assert ('loss_mssl' in losses) == bool(e['mssl']['enabled'])
            names = ('num_classes','matcher','weight_dict','eos_coef','losses','num_points','oversample_ratio',
                     'importance_sample_ratio','dn','dn_losses','panoptic_on','semantic_ce_loss')
            old = MSSLCriterion(**{k:getattr(model.criterion,k) for k in names}).cuda() if e['mssl']['enabled'] else SetCriterion(**{k:getattr(model.criterion,k) for k in names}).cuda()
            if e['mssl']['enabled']:
                old.options = model.criterion.mssl_options
            restore(captured['rng'])
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                expected = old(*captured['inputs'])
            old_after = rng_state()
            assert first_matches == observed
            for a,b in zip(after[2:],old_after[2:]):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            assert set(expected) == set(losses)
            diff = {}
            for key,value in expected.items():
                if key != 'loss_ce':
                    reference = value * old.weight_dict[key]
                    diff[key] = float((losses[key]-reference).abs())
                    torch.testing.assert_close(losses[key],reference,rtol=2e-6,atol=1e-6,msg=key)
            # Compare the Q-off native/MSSL criterion against the actual frozen
            # original source at the loss-function level in regression tests.
            assert all(p.grad is None for p in model.parameters())
            row = {'image_id':record['image_id'], 'original_single':record['mssl_source_single'],
                   'augmented_objects':len(batch[0]['instances']), 'seed':seed,
                   'quality_calls':model.criterion._q_calls, 'unchanged_weighted_loss_differences':diff,
                   'matching_rng_exact':True, 'target_stats':model.criterion.last_q_stats,
                   'changed_final_weighted_loss':float(losses['loss_ce']),
                   'native_final_weighted_loss':float(expected['loss_ce'] * old.weight_dict['loss_ce'])}
            if e['mssl']['enabled'] and record['mssl_source_single']:
                output = captured['inputs'][0]
                new_grad = torch.autograd.grad(losses['loss_mssl'],output['pred_boxes'],retain_graph=True)[0]
                old_grad = torch.autograd.grad(expected['loss_mssl'] * old.weight_dict['loss_mssl'],output['pred_boxes'],retain_graph=True)[0]
                torch.testing.assert_close(new_grad,old_grad,rtol=0,atol=0)
                row['mssl_box_gradient_exact'] = True
            report['panels'].append(row)
            mh.remove(); restore(after)
            del expected, old, losses, captured, batch
            torch.cuda.empty_cache()
        gradients = [__import__('json').loads(s) for s in (out/'weighted_class_gradients.jsonl').read_text().splitlines()]
        assert all(g['gradients']['loss']['class_head_gradient_norm'] > 0 for g in gradients)
        report['checks']['final_only_loss_routing_native_other_losses_matching_rng'] = True
        report['checks']['finite_nonzero_weighted_class_head_gradients'] = True
        entry = e['dataset']['val']; val = load_coco_json(entry['annotations'],entry['image_root'],'quality_accept_val')
        image = DatasetMapper(cfg,False)(val[0]); image.pop('instances',None)
        model.eval()
        with torch.no_grad():
            before = model([image])[0]['instances']; criterion = model.criterion; model.criterion = None
            after = model([image])[0]['instances']; model.criterion = criterion
        for key in ('pred_boxes','scores','pred_classes','pred_masks'):
            a,b=getattr(before,key),getattr(after,key)
            if key=='pred_boxes':a,b=a.tensor,b.tensor
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        report['checks']['image_only_native_inference_exact'] = True
        report['status'] = 'passed'
    except BaseException as exc:
        report.update(status='failed',error=repr(exc)); raise
    finally:
        write_json(out/'report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True);parser.add_argument('--training')
    a=parser.parse_args();run(a.config,a.output,a.training)
