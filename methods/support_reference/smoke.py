"""Cold-start finite optimization: twenty successful updates, forty-attempt cap.

Same native optimizer, schedule, mapper and AMP accumulation as formal runs.
No validation/AP evaluation; outputs cannot seed a formal run.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch

from polypbench.io import read_json, write_json, sha256
from .config import validate
from .monitor import adapter_named, native_digest, optimizer_contract, snapshot_adapter, update_record


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run(config, output):
    from methods.mssl.config import load_experiment, build_cfg
    experiment = validate(load_experiment(config))
    assert not experiment['mssl']['enabled'], 'First standalone decision stage only'
    sys.path.insert(0, experiment['repo']['path'])
    import train_net
    from detectron2.data import DatasetCatalog, build_detection_train_loader, DatasetMapper
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.utils.events import EventStorage
    from detectron2.utils.file_io import PathManager
    from methods.mssl.data import load_records, MSSLMapper
    from methods.mssl.accumulation import AccumulatingAMPTrainer
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    write_json(output/'state.json', {'status': 'preparing', 'successful_updates': 0})
    try:
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = False
        seed_all(experiment['seed'])
        cfg = build_cfg(experiment, output, device='cuda')
        model = build_model(cfg)
        # Use the exact recorded original asset, never a mature detector checkpoint.
        asset = Path(PathManager.get_local_path(cfg.MODEL.WEIGHTS))
        expected = read_json(Path(experiment['output_root'])/'maskdino-mssl-control-s42-20261018T025946/train/initialization.json')
        assert cfg.MODEL.WEIGHTS == expected['source'] and sha256(asset) == expected['sha256']
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
        initial_native = native_digest(model)
        head_initial = snapshot_adapter(model)
        optimizer = train_net.Trainer.build_optimizer(cfg, model)
        contract = optimizer_contract(model, optimizer)
        write_json(output/'optimizer_contract.json', contract)
        schedule = train_net.Trainer.build_lr_scheduler(cfg, optimizer)
        name = 'reference_smoke_train'
        train = experiment['dataset']['train']
        DatasetCatalog.register(name, lambda: load_records(train['annotations'], train['image_root'],
            name, experiment['dataset']['raw_mask_root']))
        micro = cfg.clone()
        micro.defrost()
        micro.DATASETS.TRAIN = (name,)
        micro.SOLVER.IMS_PER_BATCH = 1
        micro.freeze()
        records = []

        loader = build_detection_train_loader(micro, mapper=MSSLMapper(micro, 'annotations'))
        class RecordingLoader:
            def __iter__(self):
                for batch in loader:
                    for row in batch:
                        records.append({'image_id': row['image_id'],
                            'image_sha256': hashlib.sha256(row['image'].numpy().tobytes()).hexdigest(),
                            'boxes_sha256': hashlib.sha256(row['instances'].gt_boxes.tensor.numpy().tobytes()).hexdigest()})
                    yield batch
        model.train()
        accumulation = cfg.SOLVER.IMS_PER_BATCH
        trainer = AccumulatingAMPTrainer(model, RecordingLoader(), optimizer, accumulation)
        rows = []
        with EventStorage(0) as storage:
            for attempt in range(40):
                trainer.iter = attempt
                before = snapshot_adapter(model)
                previous = trainer.successful_updates
                trainer.run_step()
                schedule.step()
                latest = storage.latest()
                row = {'attempt': attempt+1, 'successful_updates': trainer.successful_updates,
                    'skipped': trainer.successful_updates == previous,
                    'loss_scale': trainer.grad_scaler.get_scale(),
                    'total_loss': float(latest['total_loss'][0]),
                    'adapter': update_record(model, before)}
                assert np.isfinite(row['total_loss'])
                rows.append(row)
                write_json(output/'updates.json', rows)
                write_json(output/'state.json', {'status': 'running', 'attempts': len(rows),
                    'successful_updates': trainer.successful_updates, 'elapsed_seconds': time.monotonic()-start})
                print(f"{experiment['support_reference']['variant']} attempt {attempt+1}, "
                      f"updates {trainer.successful_updates}/20, loss {row['total_loss']:.4f}", flush=True)
                storage.step()
                if trainer.successful_updates == 20:
                    break
        assert trainer.successful_updates == 20, 'Smoke did not reach 20 successful updates within cap40'
        if head_initial:
            assert any(v['unscaled_after_clip_gradient_l2'] and v['unscaled_after_clip_gradient_l2'] > 0
                for r in rows if not r['skipped'] for n, v in r['adapter'].items() if '.head.0.' in n), 'Hidden layer never received nonzero finite gradients'
            assert all(not torch.equal(p, head_initial[n]) for n, p in adapter_named(model)), 'Some adapter parameter never updated'
        assert all(torch.isfinite(p).all() for p in model.parameters()), 'Nonfinite trained parameter'
        optimizer.zero_grad(set_to_none=True)
        payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': schedule.state_dict(), 'grad_scaler': trainer.grad_scaler.state_dict(),
            'iteration': len(rows)-1, 'successful_updates': trainer.successful_updates,
            'smoke_only_no_formal_warm_start': True}
        checkpoint = output/'smoke_only.pth'
        torch.save(payload, checkpoint)
        del payload
        # Strict state + optimizer/scaler round trip, then image-only prediction parity.
        evaluation_record = copy.deepcopy(load_records(train['annotations'], train['image_root'],
            None, experiment['dataset']['raw_mask_root'])[0])
        mapped = DatasetMapper(cfg, False)(evaluation_record)
        model.eval()
        with torch.no_grad():
            expected_predictions = model([mapped])[0]['instances']
        reloaded = build_model(cfg)
        restored = torch.load(checkpoint, map_location='cpu', weights_only=False)
        reloaded.load_state_dict(restored['model'], strict=True)
        opt2 = train_net.Trainer.build_optimizer(cfg, reloaded)
        scheduler2 = train_net.Trainer.build_lr_scheduler(cfg, opt2)
        opt2.load_state_dict(restored['optimizer'])
        scheduler2.load_state_dict(restored['scheduler'])
        scaler2 = torch.amp.GradScaler('cuda')
        scaler2.load_state_dict(restored['grad_scaler'])
        assert scaler2.state_dict() == trainer.grad_scaler.state_dict()
        assert native_digest(reloaded) == native_digest(model)
        reloaded.eval()
        with torch.no_grad():
            actual_predictions = reloaded([mapped])[0]['instances']
        for a, b in ((expected_predictions.pred_boxes.tensor, actual_predictions.pred_boxes.tensor),
                     (expected_predictions.scores, actual_predictions.scores),
                     (expected_predictions.pred_masks, actual_predictions.pred_masks)):
            assert torch.equal(a, b), 'Smoke checkpoint image-only parity failed'
        write_json(output/'data_sequence.json', records)
        write_json(output/'verification.json', {'status': 'passed', 'variant': experiment['support_reference']['variant'],
            'successful_updates': 20, 'attempts': len(rows), 'amp_skips': sum(r['skipped'] for r in rows),
            'initial_native_sha256': initial_native, 'initial_asset': expected,
            'checkpoint_sha256': sha256(checkpoint), 'strict_checkpoint_reload': True,
            'optimizer_scheduler_scaler_reload': True, 'image_prediction_reload_parity': True,
            'new_parameter_count': contract['new_parameter_count'], 'microbatch': 1,
            'accumulation_steps': accumulation, 'no_ap_evaluation': True,
            'elapsed_seconds': time.monotonic()-start,
            'peak_allocated_GiB': torch.cuda.max_memory_allocated()/2**30})
        write_json(output/'state.json', {'status': 'completed', 'successful_updates': 20})
    except BaseException as error:
        write_json(output/'state.json', {'status': 'failed', 'error': repr(error), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.config, Path(args.output).resolve())
