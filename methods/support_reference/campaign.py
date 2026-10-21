"""Finite full-training queue. All cold smokes must pass before launch.

No mid-run AP-based changes and no automatic MSSL combination. Full runs never
load smoke weights. Each model is selected by Kvasir Val, then evaluated once
on the fixed PolypGen development cohort with candidate/FP diagnostics.
"""
import argparse
import copy
import datetime as dt
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import yaml

from polypbench.io import read_json, write_json, sha256
from .smoke_campaign import freeze_source, COMMON


LABELS = ('blend05', 'geometry', 'support')
CRITICAL = ('methods/support_reference/model.py', 'methods/support_reference/config.py',
    'methods/support_reference/integration.py', 'methods/support_reference/monitor.py',
    'polypbench/analysis/support_reference_prototype.py', 'methods/mssl/config.py',
    'methods/mssl/model.py', 'methods/mssl/accumulation.py', 'methods/mssl/data.py',
    'polypbench/adapters/bitmask_mapper.py', 'polypbench/adapters/instance_lsj.py')


def launch(output, smoke, preflight):
    root = Path.cwd().resolve()
    output = output.resolve()
    smoke = smoke.resolve()
    preflight = preflight.resolve()
    receipt = read_json(smoke/'verification.json')
    assert receipt['status'] == 'passed' and receipt['successful_updates_each'] == 20
    assert read_json(preflight/'verification.json')['status'] == 'passed', 'External collector preflight not passed'
    smoke_protocol = read_json(smoke/'protocol.json')
    for relative, expected in smoke_protocol['source_sha256'].items():
        assert sha256(Path(smoke_protocol['source'])/relative) == expected, 'Frozen smoke input changed: '+relative
    for relative in CRITICAL:
        assert sha256(root/relative) == smoke_protocol['source_sha256'][relative], 'Smoke-tested mechanism changed: '+relative
    if shutil.disk_usage(root).free < 15*2**30:
        raise RuntimeError('Need at least15GiB for full-run preservation')
    from methods.mssl.config import load_experiment
    from methods.mssl.run import validate_dataset, verify_repo
    from polypbench.zeroshot import validate_frozen_training
    from .config import validate
    baseline_root = root/'runs/maskdino-mssl-control-s42-20261018T025946'
    baseline, _, _ = validate_frozen_training(baseline_root)
    output.mkdir(parents=True, exist_ok=False)
    source = output/'source'
    freeze_source(root, source)
    configs, runs = {}, {}
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    for variant in LABELS:
        experiment = yaml.safe_load(Path(smoke_protocol['configs'][variant]).read_text(encoding='utf-8'))
        experiment['reference_initialization_sha256'] = receipt['shared_native_initialization_sha256']
        path = source/'configs'/f'{variant}.yaml'
        path.write_text(yaml.safe_dump(experiment, sort_keys=False), encoding='utf-8')
        resolved = validate(load_experiment(path))
        assert not resolved['mssl']['enabled'], 'Standalone full stage must keep MSSL off'
        for key in COMMON:
            assert resolved[key] == baseline[key], 'Original budget/protocol differs: '+key
        configs[variant] = str(path)
        runs[variant] = f'reference-{variant}-s42-{stamp}'
    manifest = read_json(baseline_root/'manifest.json')
    assert validate_dataset(resolved) == manifest['dataset'], 'Source data changed'
    assert verify_repo(resolved) == manifest['repo'], 'Native source changed'
    prior = read_json(root/'runs/mssl-polypgen-analysis-20261019/protocol.json')
    references = {}
    metrics, reference_diagnostics = {}, {}
    for name in ('baseline', 'mssl'):
        path = root/f'runs/initialization-intervention-full-20261021-r2/polypgen_{name}/native/evaluation.json'
        values = read_json(path)
        metrics[name] = {k: values[k] for k in ('AP', 'AP50', 'AP75', 'AP90', 'AP95')}
        reference_diagnostics[name] = {k: values[k] for k in
            ('official_fp_types', 'negative_images', 'negative_images_with_score_ge_05', 'negative_detections_score_ge_05')}
        references[str(path)] = sha256(path)
    for path in (smoke/'verification.json', preflight/'verification.json', baseline_root/'experiment.json',
                 baseline_root/'train/initialization.json', Path(prior['inputs']['annotations']),
                 Path(prior['inputs']['image_manifest'])):
        references[str(path)] = sha256(path)
    order = ['formal_unit_checks', 'existing_mssl_regression_checks']
    order += [v+'_startup_precheck' for v in LABELS]
    order += [f'{v}_{a}' for v in LABELS for a in ('train', 'source_predict', 'formal_gate')]
    order += ['target_image_hashes']
    order += [f'{v}_{a}' for v in LABELS for a in ('polypgen_inference', 'polypgen_scoring')]
    order += ['final_comparison']
    protocol = {'workspace': str(root), 'source': str(source), 'configs': configs, 'runs': runs,
        'workspace_git_commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
        'smoke': str(smoke), 'shared_native_initialization_sha256': receipt['shared_native_initialization_sha256'],
        'external_preflight': str(preflight),
        'order': order, 'reference_input_sha256': references, 'reference_metrics': metrics,
        'reference_diagnostics': reference_diagnostics,
        'annotations': prior['inputs']['annotations'], 'image_manifest': prior['inputs']['image_manifest'],
        'image_root': prior['image_root'],
        'native_stages': str(root/'runs/polypgen-candidate-stages-full-20261021/diagnostics/baseline'),
        'source_sha256': {p.relative_to(source).as_posix(): sha256(p) for p in source.rglob('*')
                          if p.is_file() and 'third_party' not in p.relative_to(source).parts},
        'budget': 'Seed42;2750 planned iterations;micro1 accumulation16;50 equivalent epochs; fresh original pretrained asset',
        'selection': 'best Kvasir Val bbox AP, earliest tie;period275;PolypGen development evaluation after all trains',
        'no_smoke_warm_start': True, 'no_intermediate_AP_tuning': True, 'no_automatic_mssl_combination': True,
        'reserved': ['PolypsSet', 'PolypDB'], 'common_fields': list(COMMON)}
    write_json(output/'protocol.json', protocol)
    write_json(output/'state.json', {'status': 'starting', 'completed': []})
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(source)+os.pathsep+baseline['repo']['path'], PYTHONUNBUFFERED='1')
    environment['PATH'] = str(Path(sys.executable).parent)+os.pathsep+environment['PATH']
    with (output/'pipeline.log').open('w', encoding='utf-8') as log:
        child = subprocess.Popen([sys.executable, '-m', 'methods.support_reference.campaign', 'run',
            '--output', str(output)], cwd=source, env=environment, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(output/'launcher.json', {'pid': child.pid})
    print({'output': str(output), 'pid': child.pid, 'order': order})


def execute(output):
    p = read_json(output/'protocol.json')
    root = Path(p['workspace'])
    state = {'status': 'running', 'controller_pid': os.getpid(), 'completed': []}
    tasks = [{'name': n, 'status': 'pending'} for n in p['order']]
    def save():
        write_json(output/'state.json', state)
        write_json(output/'queue.json', {'status': state['status'], 'tasks': tasks})
    def begin(name):
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
        save()
    def command(name, args):
        begin(name)
        with (output/(name+'.log')).open('w', encoding='utf-8') as log:
            child = subprocess.Popen([sys.executable, *args], stdout=log, stderr=subprocess.STDOUT)
            state['child_pid'] = child.pid
            save()
            try:
                code = child.wait()
            except BaseException:
                child.terminate()
                child.wait()
                raise
            if code != 0:
                raise RuntimeError(name+' failed; inspect stage log')
        done(name)
    try:
        save()
        command('formal_unit_checks', ['-m', 'unittest', 'tests.test_support_reference_training',
            'tests.test_support_reference_integration', 'tests.test_support_reference_prototype', '-v'])
        command('existing_mssl_regression_checks', ['-m', 'unittest', 'discover', '-s', 'methods/mssl/tests', '-v'])
        from methods.mssl.config import load_experiment
        for variant in LABELS:
            experiment = load_experiment(p['configs'][variant])
            experiment['reference_startup_only'] = True
            path = output/f'{variant}.startup.json'
            write_json(path, experiment)
            precheck = output/'startup'/variant
            precheck.mkdir(parents=True, exist_ok=False)
            command(variant+'_startup_precheck', ['-m', 'methods.mssl.worker', '--experiment', str(path),
                '--output', str(precheck), '--mode', 'train'])
            assert read_json(precheck/'reference_startup_gate.json')['status'] == 'passed'
        for variant in LABELS:
            trained = root/'runs'/p['runs'][variant]
            for action, label in (('train', 'train'), ('predict', 'source_predict')):
                command(variant+'_'+label, ['-m', 'methods.support_reference', action,
                    '--config', p['configs'][variant], '--run-id', p['runs'][variant]])
            name = variant+'_formal_gate'
            begin(name)
            from polypbench.zeroshot import validate_frozen_training
            validate_frozen_training(trained)
            assert read_json(trained/'train/initialization.json') == read_json(
                root/'runs/maskdino-mssl-control-s42-20261018T025946/train/initialization.json')
            assert read_json(trained/'train/reference_initialization.json')['native_parameters_sha256'] == p['shared_native_initialization_sha256']
            assert read_json(trained/'train/reference_optimizer_contract.json')['status'] == 'passed'
            done(name)
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
        from polypbench.zeroshot import validate_frozen_training, score_cohort
        for variant in LABELS:
            trained = root/'runs'/p['runs'][variant]
            _, checkpoint, digest = validate_frozen_training(trained)
            external = output/'external'/variant
            external.mkdir(parents=True, exist_ok=False)
            write_json(external/'images.json', {'images': gt['images']})
            write_json(external/'job.json', {'experiment': str(trained/'experiment.json'),
                'checkpoint': str(checkpoint), 'checkpoint_sha256': digest,
                'output': str(external), 'image_root': p['image_root'],
                'all_images': str(external/'images.json'),
                'source_predictions': str(trained/'predict_val/predictions.bbox.json')})
            command(variant+'_polypgen_inference', ['-m', 'methods.support_reference.external',
                    '--job', str(external/'job.json')])
            name = variant+'_polypgen_scoring'
            begin(name)
            score_cohort(p['annotations'], read_json(external/'predictions.all_stills.bbox.json'), external/'primary')
            done(name)
        command('final_comparison', ['-m', 'methods.support_reference.comparison', '--output', str(output)])
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
    parser.add_argument('--output', required=True)
    parser.add_argument('--smoke')
    parser.add_argument('--preflight')
    args = parser.parse_args()
    if args.action == 'launch':
        if not args.smoke or not args.preflight:
            parser.error('--smoke and --preflight are required for launch')
        launch(Path(args.output), Path(args.smoke), Path(args.preflight))
    else:
        execute(Path(args.output))
