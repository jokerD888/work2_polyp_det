"""Disposable cold-start AMP accumulation, native optimizer and strict reload."""
import argparse
import copy
import hashlib
from pathlib import Path
import sys
import time
import numpy as np
import torch
from polypbench.io import read_json, write_json, sha256
from .acceptance import seed
from .monitor import native_digest, optimizer_contract


def run(config, output, updates=20):
    from methods.mssl.config import load_experiment, build_cfg
    from .config import validate
    exp = validate(load_experiment(config))
    sys.path.insert(0, exp['repo']['path'])
    import train_net
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.data import DatasetCatalog, DatasetMapper, build_detection_train_loader
    from detectron2.utils.events import EventStorage
    from detectron2.utils.file_io import PathManager
    from methods.mssl.accumulation import AccumulatingAMPTrainer
    from .data import load_records, LocalMapper
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    seed(exp['seed'])
    cfg = build_cfg(exp, output)
    model = build_model(cfg)
    expected_asset = read_json(Path(exp['output_root'])/'mssl-official50-s42-20261018T025946/train/initialization.json')
    assert cfg.MODEL.WEIGHTS == expected_asset['source']
    assert sha256(PathManager.get_local_path(cfg.MODEL.WEIGHTS)) == expected_asset['sha256']
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    initial = native_digest(model)
    optimizer = train_net.Trainer.build_optimizer(cfg, model)
    contract = optimizer_contract(model, optimizer)
    scheduler = train_net.Trainer.build_lr_scheduler(cfg, optimizer)
    name = 'local_smoke_train'
    entry = exp['dataset']['train']
    DatasetCatalog.register(name, lambda: load_records(entry['annotations'], entry['image_root'], name,
        exp['dataset']['raw_mask_root']))
    micro = cfg.clone()
    micro.defrost()
    micro.DATASETS.TRAIN = (name,)
    micro.SOLVER.IMS_PER_BATCH = 1
    micro.freeze()
    loader = build_detection_train_loader(micro, mapper=LocalMapper(micro, 'annotations'))
    sequence = []
    class RecordLoader:
        def __iter__(self):
            for batch in loader:
                for row in batch:
                    sequence.append(dict(image_id=row['image_id'],
                        image_sha256=hashlib.sha256(row['image'].numpy().tobytes()).hexdigest(),
                        boxes_sha256=hashlib.sha256(row['instances'].gt_boxes.tensor.numpy().tobytes()).hexdigest()))
                yield batch
    model.train()
    trainer = AccumulatingAMPTrainer(model, RecordLoader(), optimizer, cfg.SOLVER.IMS_PER_BATCH)
    rows = []
    start = time.monotonic()
    with EventStorage(0) as storage:
        for attempt in range(updates+20):
            trainer.iter = attempt
            trainer.run_step()
            scheduler.step()
            stats = {k: float(v[0]) for k, v in storage.latest().items()
                     if k.startswith(('local_contrast/', 'amp/', 'optimizer/')) or k == 'total_loss'}
            assert all(np.isfinite(v) for v in stats.values())
            rows.append(dict(attempt=attempt+1, successful_updates=trainer.successful_updates, **stats))
            write_json(output/'updates.json', rows)
            write_json(output/'state.json', dict(status='running', attempts=attempt+1,
                successful_updates=trainer.successful_updates, target_updates=updates,
                elapsed_seconds=time.monotonic()-start))
            write_json(output/'data_sequence.json', sequence)
            print(f'{exp["local_contrast"]["variant"]}: {trainer.successful_updates}/{updates} successful; attempt {attempt+1}', flush=True)
            storage.step()
            if trainer.successful_updates == updates:
                break
    assert trainer.successful_updates == updates, 'Insufficient successful AMP steps'
    assert all(torch.isfinite(p).all() for p in model.parameters())
    optimizer.zero_grad(set_to_none=True)
    checkpoint = output/'smoke_only.pth'
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), grad_scaler=trainer.grad_scaler.state_dict(),
        smoke_only_no_formal_warm_start=True), checkpoint)
    model.eval()
    image = DatasetMapper(cfg, False)(copy.deepcopy(load_records(entry['annotations'], entry['image_root'],
        None, exp['dataset']['raw_mask_root'])[0]))
    with torch.no_grad():
        expected = model([image])[0]['instances']
    restored = torch.load(checkpoint, map_location='cpu', weights_only=False)
    reloaded = build_model(cfg)
    reloaded.load_state_dict(restored['model'], strict=True)
    opt2 = train_net.Trainer.build_optimizer(cfg, reloaded)
    sched2 = train_net.Trainer.build_lr_scheduler(cfg, opt2)
    opt2.load_state_dict(restored['optimizer'])
    sched2.load_state_dict(restored['scheduler'])
    scaler2 = torch.amp.GradScaler('cuda')
    scaler2.load_state_dict(restored['grad_scaler'])
    assert scaler2.state_dict() == trainer.grad_scaler.state_dict()
    assert sched2.state_dict() == scheduler.state_dict()
    assert native_digest(reloaded) == native_digest(model)
    # Adam moment tensors and counters must survive, as well as parameter groups.
    a, b = opt2.state_dict(), optimizer.state_dict()
    assert a['param_groups'] == b['param_groups']
    for key in b['state']:
        for field in b['state'][key]:
            torch.testing.assert_close(a['state'][key][field], b['state'][key][field], rtol=0, atol=0)
    reloaded.eval()
    with torch.no_grad():
        actual = reloaded([image])[0]['instances']
    for x, y in ((actual.pred_boxes.tensor, expected.pred_boxes.tensor),
                 (actual.scores, expected.scores), (actual.pred_masks, expected.pred_masks)):
        assert torch.equal(x, y), 'Strict image-only checkpoint reload failed'
    write_json(output/'verification.json', dict(status='passed', variant=exp['local_contrast']['variant'],
        initial_native_sha256=initial, successful_updates=updates, attempts=len(rows),
        optimizer=contract, strict_model_optimizer_scheduler_scaler_reload=True,
        image_prediction_reload_exact=True, no_formal_warm_start=True, no_ap_evaluation=True,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30, elapsed_seconds=time.monotonic()-start))
    write_json(output/'state.json', dict(status='completed', successful_updates=updates))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--updates', type=int, default=20)
    args = p.parse_args()
    run(args.config, args.output.resolve(), args.updates)
