"""Frozen finite queue: acceptance, both smokes, both full trainings, evaluation."""
import argparse
import copy
import datetime as dt
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import yaml
from polypbench.io import read_json, write_json, sha256


COMMON = ('dataset','overrides','model','runtime','seed','checkpoint_selection',
          'checkpoint_keep','training_budget','repo','evaluation')
LABELS = ('baseline_q','mssl_q')


def now(): return dt.datetime.now(dt.timezone.utc).isoformat()


def launch(output):
    from methods.mssl.run import verify_repo, validate_dataset
    from methods.mssl.config import load_experiment, MSSLOptions
    from dataclasses import asdict
    from .config import OPTIONS, validate
    from .loss import REFERENCE_SHA256
    from polypbench.zeroshot import validate_frozen_training
    workspace=Path.cwd().resolve()
    prior=read_json(workspace/'runs/mssl-attribution-20261018-r2/protocol.json')
    previous=workspace/'runs/mssl-polypgen-analysis-20261019'
    analysis=read_json(previous/'protocol.json')
    references={}; runs={k:prior['runs'][k] for k in ('baseline','mssl')}; originals={}
    for label,run_id in runs.items():
        trained=workspace/'runs'/run_id; validate_frozen_training(trained)
        originals[label]=read_json(trained/'experiment.json')
        for rel in ('experiment.json','manifest.json','train/selected_checkpoint.json',
                    'train/model_best.pth','train/initialization.json','predict_val/predictions.bbox.json','predict_val/metrics.json'):
            references[str(trained/rel)]=sha256(trained/rel)
        path=analysis['inputs']['predictions'][label]; references[path]=sha256(path)
    for key in COMMON:
        if originals['baseline'][key]!=originals['mssl'][key]: raise RuntimeError(f'Original A/B differ: {key}')
    target=Path(analysis['inputs']['annotations']).parent
    for path in target.glob('*.json'): references[str(path)]=sha256(path)
    for rel in ('protocol.json','uncertainty/image_multiplicities.npy','uncertainty/bootstrap_ap_vectors.npz'):
        references[str(previous/rel)]=sha256(previous/rel)
    diagnosis=workspace/'runs/mssl-quality-source-20261020/protocol.json'
    references[str(diagnosis)]=sha256(diagnosis)
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    source=out/'source';source.mkdir();ignore=shutil.ignore_patterns('__pycache__','*.pyc')
    shutil.copytree(workspace/'polypbench',source/'polypbench',ignore=ignore)
    (source/'methods').mkdir();shutil.copy2(workspace/'methods/__init__.py',source/'methods/__init__.py')
    for name in ('mssl','quality_alignment'):
        shutil.copytree(workspace/'methods'/name,source/'methods'/name,ignore=ignore)
    (source/'tests').mkdir();shutil.copy2(workspace/'tests/test_mssl_analysis.py',source/'tests/test_mssl_analysis.py')
    (source/'configs').mkdir();configs={};smokes={};stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    for label,old_label in zip(LABELS,('baseline','mssl')):
        formal=copy.deepcopy(originals[old_label]);formal.pop('source_config',None)
        formal.update(project_root='..',name=label+'_ia_bce_final_s42',purpose='fixed_2x2_quality_alignment_decision')
        formal['mssl']=asdict(MSSLOptions(**formal['mssl']).validate());formal['quality_alignment']=copy.deepcopy(OPTIONS)
        formal_path=source/'configs'/f'{label}.yaml';formal_path.write_text(yaml.safe_dump(formal,sort_keys=False),encoding='utf-8')
        loaded=validate(load_experiment(formal_path))
        for key in COMMON:
            if loaded[key]!=originals[old_label][key]: raise RuntimeError(f'Changed common protocol: {label}/{key}')
        manifest=read_json(workspace/'runs'/runs[old_label]/'manifest.json')
        if validate_dataset(loaded)!=manifest['dataset'] or verify_repo(loaded)!=manifest['repo']:
            raise RuntimeError('Original native source/data changed')
        configs[label]=str(formal_path);runs[label]=f'{label.replace("_","-")}-s42-{stamp}'
        smoke=copy.deepcopy(formal);smoke.pop('training_budget',None);smoke['name']+='_smoke'
        smoke['purpose']='runtime_gate_only_not_AP_selection'
        smoke['overrides'].update({'SOLVER.MAX_ITER':50,'SOLVER.STEPS':[],
                                  'SOLVER.CHECKPOINT_PERIOD':50,'TEST.EVAL_PERIOD':50})
        smoke_path=source/'configs'/f'{label}_smoke.yaml';smoke_path.write_text(yaml.safe_dump(smoke,sort_keys=False),encoding='utf-8')
        smokes[label]={'config':str(smoke_path),'run_id':f'{label.replace("_","-")}-smoke-{stamp}'}
    order=['quality_unit_checks','existing_mssl_regression_checks','bootstrap_regression_checks']
    order += [f'{label}_precheck' for label in LABELS]
    order += [f'{label}_smoke_{stage}' for label in LABELS for stage in ('train','predict','gate')]
    order += [f'{label}_formal_{stage}' for label in LABELS for stage in ('train','predict','checkpoint_replay')]
    order += ['validate_target_images']
    order += [f'{label}_polypgen_{stage}' for label in LABELS for stage in ('inference','scoring')]
    order += [f'{label}_source_replay' for label in LABELS]
    order += ['source_quality_diagnostics','four_group_comparison']
    reference=source/'methods/quality_alignment/reference/align_losses.py'
    if sha256(reference)!=REFERENCE_SHA256: raise RuntimeError('Author source hash mismatch')
    protocol={'created_at':now(),'workspace':str(workspace),'source_root':str(source),'runs':runs,
        'formal_configs':configs,'smokes':smokes,'order':order,'common_fields_equal_to_original':list(COMMON),
        'quality_contract':OPTIONS,'reference_sha256':REFERENCE_SHA256,
        'reference_commit':'4b125f7fe2ff8c3468cd71763fcf971d09ee38db',
        'reference_input_sha256':references,'image_root':analysis['image_root'],
        'annotations':analysis['inputs']['annotations'],'image_manifest':analysis['inputs']['image_manifest'],
        'target_root':str(target),'previous_analysis':str(previous),'previous_diagnosis':str(diagnosis),
        'source_sha256':{p.relative_to(source).as_posix():sha256(p) for p in source.rglob('*') if p.is_file()},
        'inference':'native class Top100 then class*mask quality; no additional parameters/threshold/NMS',
        'budget':'Seed42, 2750 planned iterations each, 50 equivalent epochs; same initialization, no smoke warm start',
        'selection':'source Val maximum AP, earliest tie, original period275; no external selection',
        'adaptation':'No intermediate AP-based configuration changes. Engineering failures stop the queue and require a recorded repair and renewed acceptance.',
        'attribution_limit':'IA-BCE changes soft target AND focal positive/negative loss shape/scale. No pure-ranking or new-method claim.',
        'deferred':['Quality Head','DCNv4','local contrast','PolypsSet','PolypDB','multi-seed']}
    write_json(out/'protocol.json',protocol)
    write_json(out/'queue.json',{'status':'starting','tasks':[{'name':name,'status':'pending'} for name in order]})
    env=os.environ.copy();env['PYTHONPATH']=str(source)+os.pathsep+originals['baseline']['repo']['path']
    env['PYTHONUNBUFFERED']='1';env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env['PATH']
    with (out/'pipeline.log').open('w',encoding='utf-8') as log:
        child=subprocess.Popen([sys.executable,'-m','methods.quality_alignment.campaign','run','--output',str(out)],
            cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(out/'launcher.json',{'pid':child.pid,'output':str(out),'detached':True})
    print(f'Detached IA-BCE-Final queue PID {child.pid}: {out}',flush=True)


def smoke_gate(training, mssl):
    root=Path(training)
    for stage in ('train','predict_val'):
        if read_json(root/stage/'state.json')['status']!='completed': raise RuntimeError('Incomplete smoke')
    rows=[json.loads(s) for s in (root/'train/metrics.json').read_text().splitlines() if s.strip()]
    rows=[r for r in rows if 'optimizer/successful_updates' in r and 'total_loss' in r]
    last=max(rows,key=lambda r:r['iteration'])
    if last['iteration']!=49 or last['optimizer/successful_updates']<25 or last['amp/skipped_step']!=0:
        raise RuntimeError('Insufficient successful smoke updates or unresolved final AMP overflow')
    if any(('loss_mssl' in r)!=mssl for r in rows): raise RuntimeError('Wrong MSSL route')
    stats={k:v for k,v in last.items() if k.startswith('quality/')}
    if not stats or not all(math.isfinite(v) for v in stats.values()): raise RuntimeError('Missing/nonfinite quality statistics')
    probes=[json.loads(s) for s in (root/'train/quality_gradient_probes.jsonl').read_text().splitlines()]
    if len(probes)<2 or not all(p['gradients']['loss']['class_head_gradient_norm']>0 for p in probes):
        raise RuntimeError('No finite positive class-head probe')
    return {'status':'passed','last_iteration':last['iteration'],'successful_updates':last['optimizer/successful_updates'],
        'amp_scale':last['amp/loss_scale'],'quality_stats':stats,'gradient_probes':len(probes),
        'gate':'runtime/numerics only; AP ignored','checkpoint_reload_and_full_val':True}


def execute(output, resume_training_args=None):
    out=Path(output);p=read_json(out/'protocol.json');workspace=Path(p['workspace']);source=Path(p['source_root'])
    if resume_training_args is None:
        state={'status':'running','pipeline_pid':os.getpid(),'completed':[]}
        queue={'status':'running','tasks':[{'name':name,'status':'pending'} for name in p['order']]}
    else:
        state=read_json(out/'state.json');queue=read_json(out/'queue.json')
        if state['status']!='paused' or state['stage']!='mssl_q_formal_train':
            raise RuntimeError('Continuation requires the explicitly paused MSSL+Q training stage')
        if state['completed']!=p['order'][:p['order'].index('mssl_q_formal_train')]:
            raise RuntimeError('Unexpected completed-task prefix')
        state.update(status='running',pipeline_pid=os.getpid(),resumed_at=now())
    def save():
        state['updated_at']=now();write_json(out/'state.json',state)
        queue.update(status=state['status'],updated_at=state['updated_at']);write_json(out/'queue.json',queue)
    def verify():
        for name,digest in p['source_sha256'].items():
            if sha256(source/name)!=digest: raise RuntimeError(f'Frozen source changed: {name}')
        for path,digest in p['reference_input_sha256'].items():
            if sha256(path)!=digest: raise RuntimeError(f'Reference input changed: {path}')
    def begin(label):
        if label in state['completed']: return
        verify();state.update(stage=label,command=None,child_pid=None)
        task=next(t for t in queue['tasks'] if t['name']==label);task.update(status='running',started_at=now());save()
    def done(label):
        if label in state['completed']: return
        state['completed'].append(label)
        next(t for t in queue['tasks'] if t['name']==label).update(status='completed',ended_at=now());save()
    def command(label,args):
        if label in state['completed']: return
        if label=='mssl_q_formal_train' and resume_training_args is not None:
            args=list(resume_training_args)
        begin(label);state['command']=[sys.executable,*args]
        with (out/f'{label}.log').open('w',encoding='utf-8') as log:
            child=subprocess.Popen(state['command'],cwd=source,stdout=log,stderr=subprocess.STDOUT)
            state['child_pid']=child.pid;save();code=child.wait()
        if code: raise RuntimeError(f'{label} failed ({code}); see {label}.log')
        done(label)
    def initialization_equal(training, original_label):
        expected=read_json(workspace/'runs'/p['runs'][original_label]/'train/initialization.json')
        actual=read_json(Path(training)/'train/initialization.json')
        if actual!=expected: raise RuntimeError('Initialization asset differs from original A/B')
    save()
    try:
        command('quality_unit_checks',['-m','unittest','discover','-s','methods/quality_alignment/tests','-v'])
        command('existing_mssl_regression_checks',['-m','unittest','discover','-s','methods/mssl/tests','-v'])
        command('bootstrap_regression_checks',['-m','unittest','discover','-s','tests','-p','test_mssl_analysis.py','-v'])
        for label in LABELS:
            command(f'{label}_precheck',['-m','methods.quality_alignment.precheck','--config',p['formal_configs'][label],
                '--output',str(out/f'{label}_precheck')])
            if read_json(out/f'{label}_precheck/report.json')['status']!='passed': raise RuntimeError('GPU acceptance did not pass')
        for label in LABELS:
            if all(f'{label}_smoke_{stage}' in state['completed'] for stage in ('train','predict','gate')):
                continue
            smoke=p['smokes'][label]
            for action in ('train','predict'):
                command(f'{label}_smoke_{action}',['-m','methods.quality_alignment',action,'--config',smoke['config'],'--run-id',smoke['run_id']])
            stage=f'{label}_smoke_gate';begin(stage)
            initialization_equal(workspace/'runs'/smoke['run_id'],'mssl' if label=='mssl_q' else 'baseline')
            write_json(out/f'{stage}.json',smoke_gate(workspace/'runs'/smoke['run_id'],label=='mssl_q'));done(stage)
        # Both complete smoke gates precede either formal training. Never branch on AP.
        for label in LABELS:
            trained=workspace/'runs'/p['runs'][label]
            for action in ('train','predict'):
                command(f'{label}_formal_{action}',['-m','methods.quality_alignment',action,'--config',p['formal_configs'][label],
                    '--run-id',p['runs'][label]])
            command(f'{label}_formal_checkpoint_replay',['-m','methods.quality_alignment.precheck','--config',p['formal_configs'][label],
                '--training',str(trained),'--output',str(out/f'{label}_checkpoint_replay')])
            initialization_equal(trained,'mssl' if label=='mssl_q' else 'baseline')
        begin('validate_target_images');gt=read_json(p['annotations']);names={im['file_name'] for im in gt['images']};verified=set()
        for image in read_json(p['image_manifest']):
            if image['relative_path'] in names:
                if sha256(Path(p['image_root'])/image['relative_path'])!=image['sha256']: raise RuntimeError('Target image bytes changed')
                verified.add(image['relative_path'])
        if verified!=names: raise RuntimeError('Target manifest coverage incomplete')
        done('validate_target_images')
        from polypbench.zeroshot import validate_frozen_training, score_cohort, COHORTS
        for label in LABELS:
            trained=workspace/'runs'/p['runs'][label];_,checkpoint,digest=validate_frozen_training(trained)
            external=out/'external'/label;external.mkdir(parents=True,exist_ok=False)
            write_json(external/'primary.images.json',{'images':gt['images']})
            write_json(external/'job.json',{'experiment':str(trained/'experiment.json'),'checkpoint':str(checkpoint),
                'checkpoint_sha256':digest,'output':str(external),'image_root':p['image_root'],
                'all_images':str(external/'primary.images.json'),'source_predictions':str(trained/'predict_val/predictions.bbox.json')})
            command(f'{label}_polypgen_inference',['-m','polypbench.adapters.zeroshot_worker','--job',str(external/'job.json')])
            stage=f'{label}_polypgen_scoring';begin(stage);predictions=read_json(external/'predictions.all_stills.bbox.json')
            for name,filename in COHORTS.items(): score_cohort(Path(p['target_root'])/filename,predictions,external/name)
            done(stage)
        # Apply the same pre-existing read-only source diagnosis to both new checkpoints.
        diagnosis=out/'source_quality_diagnostics';diagnosis.mkdir(exist_ok=False)
        previous_diagnosis=read_json(p['previous_diagnosis'])
        diagnostic_protocol={key:copy.deepcopy(previous_diagnosis[key]) for key in ('annotations','image_root','rules')}
        diagnostic_protocol.update(workspace=str(workspace),source_root=str(source),training={},input_sha256={})
        source_gt=read_json(diagnostic_protocol['annotations'])
        diagnostic_protocol['input_sha256'][diagnostic_protocol['annotations']]=sha256(diagnostic_protocol['annotations'])
        for image in source_gt['images']:
            path=str(Path(diagnostic_protocol['image_root'])/image['file_name'])
            diagnostic_protocol['input_sha256'][path]=sha256(path)
        for label in LABELS:
            trained=workspace/'runs'/p['runs'][label];e,checkpoint,digest=validate_frozen_training(trained)
            job={'experiment':str(trained/'experiment.json'),'checkpoint':str(checkpoint),'checkpoint_sha256':digest,
                 'annotations':e['dataset']['val']['annotations'],'image_root':e['dataset']['val']['image_root'],
                 'predictions':str(trained/'predict_val/predictions.bbox.json'),'output':str(diagnosis/'score_replay'/label)}
            write_json(diagnosis/f'{label}.job.json',job)
            diagnostic_protocol['training'][label]={'run_root':str(trained),**job}
            for path in (job['experiment'],job['checkpoint'],job['predictions'],str(trained/'predict_val/metrics.json')):
                diagnostic_protocol['input_sha256'][path]=sha256(path)
            command(f'{label}_source_replay',['-m','polypbench.analysis.score_replay','--job',str(diagnosis/f'{label}.job.json')])
        diagnostic_protocol.update(created_at=now(),purpose='fixed post-training source diagnosis; not adaptive method selection',
                                   source_sha256=p['source_sha256'])
        write_json(diagnosis/'protocol.json',diagnostic_protocol)
        command('source_quality_diagnostics',['-m','polypbench.analysis.quality_study','analyze','--output',str(diagnosis)])
        command('four_group_comparison',['-m','methods.quality_alignment.comparison','--output',str(out)])
        if state['completed']!=p['order']: raise RuntimeError('Queue execution order drift')
        state['status']='completed'
    except BaseException as error:
        state.update(status='failed',error=repr(error))
        for task in queue['tasks']:
            if task['status']=='running':task.update(status='failed',error=repr(error),ended_at=now())
        raise
    finally: save()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('launch','run'));parser.add_argument('--output',required=True)
    a=parser.parse_args();(launch if a.action=='launch' else execute)(a.output)
