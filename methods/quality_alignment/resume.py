"""Audited process continuation; frozen detector/loss sources remain unchanged."""
import argparse
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from polypbench.io import read_json, write_json, sha256


def stamp(): return dt.datetime.now(dt.timezone.utc).isoformat()


def validate_training(training):
    from methods.mssl.run import adapter_sources, validate_dataset, verify_repo
    from polypbench.config import fingerprint
    training=Path(training);e=read_json(training/'experiment.json');manifest=read_json(training/'manifest.json')
    if fingerprint(e)!=manifest['config_fingerprint']:raise RuntimeError('Training configuration changed')
    if adapter_sources(e)!=read_json(training/'adapter_sources.json'):raise RuntimeError('Frozen training adapter changed')
    if validate_dataset(e)!=manifest['dataset']:raise RuntimeError('Training dataset changed')
    if verify_repo(e)!=manifest['repo']:raise RuntimeError('Native model repository changed')
    return {'status':'passed','config_fingerprint':manifest['config_fingerprint'],'data_and_native_repo_equal':True}


def verify(output):
    out=Path(output);p=read_json(out/'protocol.json');source=Path(p['source_root'])
    for rel,digest in p['source_sha256'].items():
        if sha256(source/rel)!=digest: raise RuntimeError(f'Frozen source changed: {rel}')
    for filename,digest in p['reference_input_sha256'].items():
        if sha256(filename)!=digest: raise RuntimeError(f'Reference changed: {filename}')
    return p


def launch(output):
    out=Path(output).resolve();p=verify(out);state=read_json(out/'state.json')
    if state['status']!='paused':raise RuntimeError('Queue is not paused')
    trained=Path(p['workspace'])/'runs'/p['runs']['mssl_q'];checkpoint=trained/'train/model_0000549.pth'
    training_validation=validate_training(trained)
    import torch
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['iteration']!=549 or saved['trainer']['iteration']!=549:raise RuntimeError('Unexpected checkpoint iteration')
    inner=saved['trainer']['_trainer']
    if not {'optimizer','grad_scaler'}<=inner.keys() or 'LRScheduler' not in saved['trainer']['hooks']:
        raise RuntimeError('Incomplete optimizer/AMP/scheduler state')
    artifact=out/('resume_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));artifact.mkdir(exist_ok=False)
    workspace=Path(p['workspace'])
    shutil.copy2(Path(__file__),artifact/'resume.py')
    shutil.copy2(workspace/'methods/quality_alignment/campaign.py',artifact/'campaign.py')
    shutil.copy2(checkpoint,artifact/checkpoint.name)
    if sha256(artifact/checkpoint.name)!=sha256(checkpoint):raise RuntimeError('Checkpoint archive mismatch')
    for filename in ('state.json','queue.json','launcher.json'):
        shutil.copy2(out/filename,artifact/('previous_'+filename))
    previous_log=out/'mssl_q_formal_train.log'
    if previous_log.is_file():shutil.copy2(previous_log,artifact/'before_mssl_q_formal_train.log')
    for filename in ('environment.json','method.json','initialization.json'):
        path=trained/'train'/filename
        if path.is_file():shutil.copy2(path,artifact/('before_'+filename))
    for path in (trained/'train/metrics.json',trained/'train/quality_gradient_probes.jsonl'):
        shutil.copy2(path,artifact/('before_'+path.name))
    record={'created_at':stamp(),'original_protocol_sha256':sha256(out/'protocol.json'),
        'checkpoint':str(artifact/checkpoint.name),'checkpoint_sha256':sha256(artifact/checkpoint.name),
        'checkpoint_iteration':549,'next_iteration':550,'planned_max_iteration':2750,
        'training_root':str(trained),'source_root':p['source_root'],
        'controller_sha256':{name:sha256(artifact/name) for name in ('resume.py','campaign.py')},
        'restore':['model','optimizer','scheduler','AMP scaler','successful optimizer updates','historical best Val selection'],
        'data_position':'Replay/discard first 550*16 mapped microbatches on the reconstructed native loader; no model updates.',
        'limitation':'Original checkpoint omits Python/NumPy/Torch/CUDA RNG and worker states. Initialization and data cursor are reconstructed, but continuation is not guaranteed bitwise equal to uninterrupted training. Record this interruption in research interpretation.',
        'method_change':False,'retuned_after_AP':False}
    write_json(artifact/'protocol.json',record)
    write_json(artifact/'training_validation.json',training_validation)
    env=os.environ.copy();env['PYTHONPATH']=p['source_root']+os.pathsep+read_json(trained/'experiment.json')['repo']['path']
    env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env['PATH'];env['PYTHONUNBUFFERED']='1'
    with (artifact/'pipeline.log').open('w',encoding='utf-8') as log:
        child=subprocess.Popen([sys.executable,str(artifact/'resume.py'),'run','--output',str(out),'--artifact',str(artifact)],
            cwd=p['source_root'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/'launcher.json',{'pid':child.pid,'output':str(out),'detached':True,'resume_artifact':str(artifact)})
    write_json(out/'resume_record.json',{**record,'artifact':str(artifact),'pipeline_pid':child.pid})
    print(f'Resumed finite queue PID {child.pid}: {artifact}',flush=True)


def run(output,artifact):
    p=verify(output);artifact=Path(artifact);r=read_json(artifact/'protocol.json')
    for name,digest in r['controller_sha256'].items():
        if sha256(artifact/name)!=digest:raise RuntimeError('Resume controller changed')
    spec=importlib.util.spec_from_file_location('audited_quality_continuation',artifact/'campaign.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.execute(output,resume_training_args=[str(artifact/'resume.py'),'worker','--output',str(output),'--artifact',str(artifact)])


def rollback_logs(training,artifact,next_iteration):
    for filename,field in (('metrics.json','iteration'),('quality_gradient_probes.jsonl','iteration')):
        path=training/filename
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        kept=[row for row in rows if row[field]<next_iteration or
              (filename=='metrics.json' and row[field]==next_iteration and 'bbox/AP' in row
               and 'optimizer/successful_updates' not in row)]
        path.write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in kept),encoding='utf-8')
    write_json(artifact/'log_rollback.json',{'status':'completed','next_iteration':next_iteration,
        'old_full_logs_preserved':True,'reason':'Discard observations after the restored checkpoint; avoid duplicate iterations/selected gradients.'})


def restore_trainer(trainer,checkpoint,best,artifact):
    """Restore actual native checkpointables without changing training losses."""
    import torch
    from detectron2.engine.hooks import BestCheckpointer
    trainer.checkpointer.load(str(checkpoint))
    trainer.start_iter=trainer.iter+1
    if trainer.start_iter!=550:raise RuntimeError('Wrong restored iteration')
    named=[(name,param) for name,param in trainer.model.named_parameters() if 'class_embed' in name and param.requires_grad]
    steps=[int(trainer.optimizer.state[param]['step'].item()) for _,param in named]
    if not steps or len(set(steps))!=1:raise RuntimeError('Ambiguous classification-head optimizer step count')
    trainer._trainer.successful_updates=steps[0]
    hooks=[h for h in trainer._hooks if isinstance(h,BestCheckpointer)]
    if len(hooks)!=1:raise RuntimeError('Unexpected best-checkpoint hooks')
    hooks[0].best_metric=best['metric_percent'];hooks[0].best_iter=best['iteration']
    # The original native loader is reconstructed under the same Seed42 setup.
    # Advancing mapped batches avoids restarting the 50-epoch data cursor at 0.
    discarded=trainer.start_iter*trainer._trainer.accumulation_steps
    for index in range(discarded):
        next(trainer._trainer._data_loader_iter)
        if (index+1)%1000==0:
            write_json(artifact/'restore_progress.json',{'stage':'advance_native_data_cursor','microbatches_done':index+1,'total':discarded})
    row={'status':'restored','start_iteration':trainer.start_iter,'successful_updates':steps[0],
        'grad_scaler':trainer._trainer.grad_scaler.state_dict(),'scheduler':trainer.scheduler.state_dict(),
        'optimizer_lrs':[group['lr'] for group in trainer.optimizer.param_groups],
        'historical_best':best,'discarded_mapped_microbatches':discarded,'parameter_gradients_empty':all(p.grad is None for p in trainer.model.parameters()),
        'checkpoint_sha256':sha256(checkpoint),'restored_at':stamp()}
    if not row['parameter_gradients_empty']:raise RuntimeError('Data cursor replay altered gradients')
    write_json(artifact/'restored.json',row)
    print(f'Restored iteration {trainer.start_iter}, successful updates {steps[0]}, AMP scale {row["grad_scaler"]["scale"]}; data cursor advanced {discarded} microbatches.',flush=True)


def worker(output,artifact):
    p=verify(output);artifact=Path(artifact);record=read_json(artifact/'protocol.json')
    trained=Path(record['training_root']);training=trained/'train';e=read_json(trained/'experiment.json')
    checkpoint=Path(record['checkpoint'])
    if sha256(checkpoint)!=record['checkpoint_sha256']:raise RuntimeError('Resume checkpoint changed')
    best=read_json(training/'best_checkpoint.json')
    rollback_logs(training,artifact,record['next_iteration'])
    write_json(training/'state.json',{'status':'running','mode':'train','resumed_at':stamp(),'from_iteration':549,'resume_artifact':str(artifact)})
    sys.path.insert(0,e['repo']['path'])
    import train_net
    train_net.Trainer.resume_or_load=lambda self,resume=False:restore_trainer(self,checkpoint,best,artifact)
    from methods.mssl import worker as frozen_worker
    sys.argv=[str(Path(__file__)),'--experiment',str(trained/'experiment.json'),'--output',str(training),'--mode','train','--split','val']
    try:
        frozen_worker.main()
        final=training/'model_final.pth';selected=training/'model_best.pth'
        if not final.is_file() or not selected.is_file():raise RuntimeError('Incomplete resumed training output')
        write_json(training/'checkpoint.json',{'path':str(final),'sha256':sha256(final),'selection':'final'})
        write_json(training/'selected_checkpoint.json',{'path':str(selected),'sha256':sha256(selected),'selection':'best_val_ap',
            'validation':read_json(training/'best_checkpoint.json')})
        write_json(training/'state.json',{'status':'completed','mode':'train','finished_at':stamp(),'resume_artifact':str(artifact)})
    except BaseException as exc:
        write_json(training/'state.json',{'status':'failed','mode':'train','error':repr(exc),'resume_artifact':str(artifact)});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('launch','run','worker'))
    parser.add_argument('--output',required=True);parser.add_argument('--artifact');a=parser.parse_args()
    if a.action=='launch':launch(a.output)
    elif a.action=='run':run(a.output,a.artifact)
    else:worker(a.output,a.artifact)
