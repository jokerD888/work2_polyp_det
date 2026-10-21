"""Freeze and execute a finite four-condition cold-start smoke queue."""
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
from .config import VARIANTS


COMMON = ('dataset', 'overrides', 'model', 'runtime', 'seed', 'checkpoint_selection',
          'checkpoint_keep', 'training_budget', 'repo', 'evaluation')


def freeze_source(root, source):
    source.mkdir()
    for name in ('methods', 'polypbench', 'tests'):
        shutil.copytree(root/name, source/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (source/'third_party').mkdir()
    os.symlink(root/'third_party/MaskDINO', source/'third_party/MaskDINO', target_is_directory=True)
    (source/'configs').mkdir()


def launch(output):
    root = Path.cwd().resolve()
    output = output.resolve()
    if shutil.disk_usage(root).free < 15*2**30:
        raise RuntimeError('Need at least15GiB to preserve independent smoke/checkpoints')
    output.mkdir(parents=True, exist_ok=False)
    source = output/'source'
    freeze_source(root, source)
    baseline = read_json(root/'runs/maskdino-mssl-control-s42-20261018T025946/experiment.json')
    baseline.pop('source_config', None)
    from dataclasses import asdict
    from methods.mssl.config import MSSLOptions, load_experiment
    from .config import validate
    baseline['mssl'] = asdict(MSSLOptions(**baseline['mssl']).validate())
    configs = {}
    for variant in VARIANTS:
        experiment = copy.deepcopy(baseline)
        experiment.update(project_root='..', name='reference_'+variant+'_s42',
            purpose='fixed_baseline_reference_initialization_decision')
        experiment['support_reference'] = {'variant': variant}
        path = source/'configs'/f'{variant}.yaml'
        path.write_text(yaml.safe_dump(experiment, sort_keys=False), encoding='utf-8')
        loaded = validate(load_experiment(path))
        for key in COMMON:
            if loaded[key] != baseline[key]:
                raise RuntimeError('Original common protocol changed: '+key)
        configs[variant] = str(path)
    protocol = {'workspace': str(root), 'source': str(source), 'configs': configs,
        'variants': list(VARIANTS), 'successful_updates_each': 20, 'max_attempts_each': 40,
        'microbatch': 1, 'effective_batch': 16, 'seed': 42,
        'ap_evaluation': False, 'formal_warm_start': False,
        'formal_common_fields': list(COMMON),
        'source_sha256': {p.relative_to(source).as_posix(): sha256(p) for p in source.rglob('*')
                          if p.is_file() and 'third_party' not in p.relative_to(source).parts},
        'created_at': dt.datetime.now(dt.timezone.utc).isoformat()}
    write_json(output/'protocol.json', protocol)
    write_json(output/'state.json', {'status': 'starting', 'completed': []})
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(source)+os.pathsep+baseline['repo']['path'], PYTHONUNBUFFERED='1')
    environment['PATH'] = str(Path(sys.executable).parent)+os.pathsep+environment['PATH']
    with (output/'pipeline.log').open('w', encoding='utf-8') as log:
        child = subprocess.Popen([sys.executable, '-m', 'methods.support_reference.smoke_campaign',
            'run', '--output', str(output)], cwd=source, env=environment,
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    write_json(output/'launcher.json', {'pid': child.pid})
    print({'output': str(output), 'pid': child.pid})


def execute(output):
    protocol = read_json(output/'protocol.json')
    state = {'status': 'running', 'controller_pid': os.getpid(), 'completed': []}
    tasks = [{'name': 'cpu_tests', 'status': 'pending'}] + [
        {'name': v+'_smoke', 'status': 'pending'} for v in VARIANTS] + [{'name': 'common_start_gate', 'status': 'pending'}]
    def save():
        write_json(output/'state.json', state)
        write_json(output/'queue.json', {'status': state['status'], 'tasks': tasks})
    def command(index, args):
        task = tasks[index]
        state['stage'] = task['name']
        task['status'] = 'running'
        save()
        with (output/(task['name']+'.log')).open('w', encoding='utf-8') as log:
            child = subprocess.Popen([sys.executable, *args], stdout=log, stderr=subprocess.STDOUT)
            state['child_pid'] = child.pid
            save()
            if child.wait() != 0:
                raise RuntimeError(task['name']+' failed')
        task['status'] = 'completed'
        state['completed'].append(task['name'])
        save()
    try:
        save()
        command(0, ['-m', 'unittest', 'tests.test_support_reference_prototype',
            'tests.test_support_reference_integration', 'tests.test_support_reference_training', '-v'])
        for i, variant in enumerate(VARIANTS, 1):
            command(i, ['-m', 'methods.support_reference.smoke', '--config', protocol['configs'][variant],
                        '--output', str(output/variant)])
        tasks[-1]['status'] = 'running'
        state['stage'] = tasks[-1]['name']
        save()
        receipts = [read_json(output/v/'verification.json') for v in VARIANTS]
        assert all(r['status'] == 'passed' and r['successful_updates'] == 20 for r in receipts)
        assert len({r['initial_native_sha256'] for r in receipts}) == 1, 'Native cold initializations differ'
        sequences = [read_json(output/v/'data_sequence.json') for v in VARIANTS]
        count = min(map(len, sequences))
        assert all(s[:count] == sequences[0][:count] for s in sequences[1:]), 'Augmentation/sampler prefix differs'
        for relative, expected in protocol['source_sha256'].items():
            assert sha256(Path(protocol['source'])/relative) == expected, 'Frozen smoke source changed'
        write_json(output/'verification.json', {'status': 'passed', 'variants': list(VARIANTS),
            'successful_updates_each': 20, 'shared_native_initialization_sha256': receipts[0]['initial_native_sha256'],
            'identical_image_and_box_augmentation_prefix_microbatches': count,
            'no_ap_evaluation': True, 'no_formal_warm_start': True,
            'receipts': receipts})
        tasks[-1]['status'] = 'completed'
        state['completed'].append(tasks[-1]['name'])
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
    args = parser.parse_args()
    (launch if args.action == 'launch' else execute)(Path(args.output))
