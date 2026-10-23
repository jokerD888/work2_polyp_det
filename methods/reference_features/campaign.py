"""Finite H1/H2 campaign: acceptance -> cold smoke -> startup -> full training.

Same seed/budget/data/selection as Baseline. No mid-run AP tuning, no automatic
MSSL combination. Stops on engineering failure; preserves logs and snapshots.
"""
import argparse
import copy
from dataclasses import asdict
import datetime as dt
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import yaml
from polypbench.io import read_json, write_json, sha256
from methods.support_reference.smoke_campaign import freeze_source, COMMON

VARIANTS = ('mask_extra', 'encoder_extra')
SMOKES = ('native', *VARIANTS)


def launch(output, acceptance):
    root = Path.cwd().resolve()
    output, acceptance = output.resolve(), acceptance.resolve()
    assert read_json(acceptance/'verification.json')['status'] == 'passed'
    if shutil.disk_usage(root).free < 15*2**30:
        raise RuntimeError('Need at least 15GiB free')
    from methods.mssl.config import load_experiment, MSSLOptions
    from polypbench.zeroshot import validate_frozen_training
    from .config import validate
    baseline_root = root/'runs/maskdino-mssl-control-s42-20261018T025946'
    baseline, _, _ = validate_frozen_training(baseline_root)
    base = copy.deepcopy(baseline)
    base.pop('source_config', None)
    base['mssl'] = asdict(MSSLOptions(**base['mssl']).validate())
    base['project_root'] = '..'
    base['reference_initialization_sha256'] = read_json(
        root/'runs/support-reference-cold-smoke-20261021/verification.json')['shared_native_initialization_sha256']
    output.mkdir(parents=True, exist_ok=False)
    source = output/'source'
    freeze_source(root, source)
    configs, runs = {}, {}
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    for variant in SMOKES:
        experiment = copy.deepcopy(base)
        experiment.update(name='features_'+variant+'_s42', purpose='first_layer_extra_reference_features_decision',
            reference_features={'variant': variant})
        path = source/'configs'/f'{variant}.yaml'
        path.write_text(yaml.safe_dump(experiment, sort_keys=False))
        resolved = validate(load_experiment(path))
        for field in COMMON:
            assert resolved[field] == baseline[field], 'Changed common field: '+field
        configs[variant] = str(path)
        runs[variant] = 'features-'+variant+'-s42-'+stamp
    parent = read_json(root/'runs/support-reference-recovery-20261022/protocol.json')
    reference_files = dict(parent['reference_input_sha256'])
    reference_files[str(acceptance/'verification.json')] = sha256(acceptance/'verification.json')
    order = ['unit_and_regression_checks', 'existing_mssl_regression_checks'] + [v+'_cold_smoke' for v in SMOKES]
    order += ['common_smoke_gate', 'collector_preflight'] + [v+'_startup_precheck' for v in VARIANTS]
    order += [v+'_'+a for v in VARIANTS for a in ('train', 'source_predict', 'formal_gate')]
    order += ['target_image_hashes']
    order += [v+'_'+a for v in VARIANTS for a in ('polypgen_inference', 'polypgen_scoring')]
    order += ['final_comparison']
    p = {k: parent[k] for k in ('annotations', 'image_manifest', 'image_root', 'native_stages',
        'reference_metrics', 'reference_diagnostics')}
    p.update(workspace=str(root), source=str(source), configs=configs, runs=runs,
        variants=list(VARIANTS), smoke_variants=list(SMOKES), order=order,
        acceptance=str(acceptance), reference_input_sha256=reference_files,
        shared_native_initialization_sha256=base['reference_initialization_sha256'],
        source_sha256={p.relative_to(source).as_posix(): sha256(p) for p in source.rglob('*')
            if p.is_file() and 'third_party' not in p.relative_to(source).parts},
        budget='Seed42;2750 planned iterations;micro1 accumulation16;50 equivalent epochs; fresh original ImageNet R50',
        selection='best Kvasir Val bbox AP, earliest tie, eval period275; PolypGen only after both trains',
        no_smoke_warm_start=True, no_intermediate_AP_tuning=True, no_automatic_mssl_combination=True,
        reserved=['PolypsSet', 'PolypDB'], created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        treatment='Only extra sampler reference differs; same extra sampler+projection, native geometry unmodified',
        git_commit=subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip())
    write_json(output/'protocol.json', p)
    write_json(output/'state.json', {'status': 'starting', 'completed': []})
    env = os.environ.copy()
    env.update(PYTHONPATH=str(source)+os.pathsep+baseline['repo']['path'], PYTHONUNBUFFERED='1')
    env['PATH'] = str(Path(sys.executable).parent)+os.pathsep+env['PATH']
    with (output/'pipeline.log').open('w') as log:
        child = subprocess.Popen([sys.executable, '-m', 'methods.reference_features.campaign', 'run',
            '--output', str(output)], cwd=source, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(output/'launcher.json', {'pid': child.pid})
    print({'output': str(output), 'pid': child.pid, 'order': order})


def execute(output):
    p = read_json(output/'protocol.json')
    root = Path(p['workspace'])
    state = {'status': 'running', 'controller_pid': os.getpid(), 'completed': []}
    tasks = [{'name': n, 'status': 'pending'} for n in p['order']]
    def save():
        state['updated_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(output/'state.json', state)
        write_json(output/'queue.json', {'status': state['status'], 'tasks': tasks})
    def begin(name):
        assert name == p['order'][len(state['completed'])]
        for relative, expected in p['source_sha256'].items():
            assert sha256(Path(p['source'])/relative) == expected, 'Frozen source changed: '+relative
        for path, expected in p['reference_input_sha256'].items():
            assert sha256(path) == expected, 'Reference changed: '+path
        state['stage'] = name
        next(t for t in tasks if t['name'] == name)['status'] = 'running'
        save()
    def done(name):
        next(t for t in tasks if t['name'] == name)['status'] = 'completed'
        state['completed'].append(name)
        state.pop('child_pid', None)
        save()
    def command(name, args):
        begin(name)
        with (output/(name+'.log')).open('w') as log:
            child = subprocess.Popen([sys.executable, *args], stdout=log, stderr=subprocess.STDOUT)
            state['child_pid'] = child.pid
            save()
            try:
                code = child.wait()
            except BaseException:
                child.terminate()
                child.wait()
                raise
            if code:
                raise RuntimeError(name+' failed; see task log')
        done(name)
    try:
        save()
        command('unit_and_regression_checks', ['-m', 'unittest', 'tests.test_reference_features',
            'tests.test_support_reference_training', 'tests.test_support_reference_integration',
            'tests.test_support_reference_prototype', '-v'])
        command('existing_mssl_regression_checks', ['-m', 'unittest', 'discover', '-s', 'methods/mssl/tests', '-v'])
        for variant in SMOKES:
            command(variant+'_cold_smoke', ['-m', 'methods.reference_features.smoke',
                '--config', p['configs'][variant], '--output', str(output/'smoke'/variant)])
        begin('common_smoke_gate')
        receipts = [read_json(output/'smoke'/v/'verification.json') for v in SMOKES]
        assert all(r['status'] == 'passed' and r['successful_updates'] == 20 for r in receipts)
        assert all(r['initial_native_sha256'] == p['shared_native_initialization_sha256'] for r in receipts)
        assert receipts[1]['new_parameter_count'] == receipts[2]['new_parameter_count'] > 0
        sequences = [read_json(output/'smoke'/v/'data_sequence.json') for v in SMOKES]
        count = min(map(len, sequences))
        assert all(s[:count] == sequences[0][:count] for s in sequences[1:]), 'Sampler/augmentation prefix differs'
        write_json(output/'smoke/protocol.json', p)
        write_json(output/'smoke/verification.json', {'status': 'passed', 'receipts': receipts,
            'identical_image_box_prefix_microbatches': count, 'no_formal_warm_start': True})
        done('common_smoke_gate')
        command('collector_preflight', ['-m', 'methods.reference_features.preflight',
            '--smoke', str(output/'smoke'), '--output', str(output/'collector_preflight')])
        from methods.mssl.config import load_experiment
        for variant in VARIANTS:
            experiment = load_experiment(p['configs'][variant])
            experiment['reference_startup_only'] = True
            path = output/(variant+'.startup.json')
            write_json(path, experiment)
            precheck = output/'startup'/variant
            precheck.mkdir(parents=True, exist_ok=False)
            command(variant+'_startup_precheck', ['-m', 'methods.mssl.worker', '--experiment', str(path),
                '--output', str(precheck), '--mode', 'train'])
            assert read_json(precheck/'reference_startup_gate.json')['status'] == 'passed'
        from polypbench.zeroshot import validate_frozen_training, score_cohort
        for variant in VARIANTS:
            trained = root/'runs'/p['runs'][variant]
            for action, label in (('train', 'train'), ('predict', 'source_predict')):
                command(variant+'_'+label, ['-m', 'methods.reference_features', action,
                    '--config', p['configs'][variant], '--run-id', p['runs'][variant]])
            begin(variant+'_formal_gate')
            validate_frozen_training(trained)
            assert read_json(trained/'train/initialization.json') == read_json(
                root/'runs/maskdino-mssl-control-s42-20261018T025946/train/initialization.json')
            assert read_json(trained/'train/reference_initialization.json')['native_parameters_sha256'] == p['shared_native_initialization_sha256']
            done(variant+'_formal_gate')
        begin('target_image_hashes')
        gt = read_json(p['annotations'])
        names = {im['file_name'] for im in gt['images']}
        verified = set()
        for image in read_json(p['image_manifest']):
            if image['relative_path'] in names:
                assert sha256(Path(p['image_root'])/image['relative_path']) == image['sha256']
                verified.add(image['relative_path'])
        assert verified == names and len(names) == 1473
        done('target_image_hashes')
        for variant in VARIANTS:
            trained = root/'runs'/p['runs'][variant]
            _, checkpoint, digest = validate_frozen_training(trained)
            external = output/'external'/variant
            external.mkdir(parents=True, exist_ok=False)
            write_json(external/'images.json', {'images': gt['images']})
            write_json(external/'job.json', {'experiment': str(trained/'experiment.json'),
                'checkpoint': str(checkpoint), 'checkpoint_sha256': digest, 'output': str(external),
                'image_root': p['image_root'], 'all_images': str(external/'images.json'),
                'source_predictions': str(trained/'predict_val/predictions.bbox.json')})
            command(variant+'_polypgen_inference', ['-m', 'methods.reference_features.external', '--job', str(external/'job.json')])
            begin(variant+'_polypgen_scoring')
            score_cohort(p['annotations'], read_json(external/'predictions.all_stills.bbox.json'), external/'primary')
            done(variant+'_polypgen_scoring')
        command('final_comparison', ['-m', 'methods.reference_features.comparison', '--output', str(output)])
        assert state['completed'] == p['order']
        state['status'] = 'completed'
    except BaseException as error:
        state.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        for task in tasks:
            if task['status'] == 'running':
                task['status'] = 'failed'
        raise
    finally:
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('launch', 'run'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--acceptance', type=Path)
    args = parser.parse_args()
    if args.action == 'launch':
        if args.acceptance is None:
            parser.error('--acceptance is required')
        launch(args.output, args.acceptance)
    else:
        execute(args.output)
