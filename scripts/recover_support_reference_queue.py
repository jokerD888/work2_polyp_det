"""Recover one interrupted support run without changing frozen experiment code.

The incomplete run is retained and excluded, not resumed: native checkpoints
do not contain the full augmentation/RNG state. Completed controls are reused.
"""
import argparse
import copy
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def derive_protocol(protocol, state, original, stamp):
    prefix = protocol['order'][:protocol['order'].index('support_train')]
    if state.get('stage') != 'support_train' or state.get('completed') != prefix:
        raise ValueError('Recovery only supports the interrupted third training stage')
    result = copy.deepcopy(protocol)
    result['runs']['support'] += '-restart-' + stamp
    result['recovery'] = {
        'original_queue': str(original), 'interrupted_run': protocol['runs']['support'],
        'completed': list(prefix), 'policy': 'fresh same-seed same-budget restart of support only',
        'reason': 'Prior controller and training processes absent; native checkpoint lacks complete RNG/augmentation state',
        'no_partial_checkpoint_selection': True, 'configuration_changes': False,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    original, output = args.original.resolve(), args.output.resolve()
    raw = json.loads((original/'protocol.json').read_text())
    # All model, training and evaluation imports come from the original snapshot.
    sys.path.insert(0, raw['source'])
    from polypbench.io import read_json, write_json, sha256
    from polypbench.zeroshot import validate_frozen_training, score_cohort

    def verify(p):
        for name, digest in p['source_sha256'].items():
            if sha256(Path(p['source'])/name) != digest:
                raise RuntimeError('Frozen source changed: ' + name)
        for name, digest in p['reference_input_sha256'].items():
            if sha256(name) != digest:
                raise RuntimeError('Reference changed: ' + name)

    if not args.run:
        previous = read_json(original/'state.json')
        for key in ('controller_pid', 'child_pid'):
            if Path('/proc', str(previous.get(key, -1))).exists():
                raise RuntimeError('Old PID still exists; investigate before recovery')
        verify(raw)
        for variant in ('blend05', 'geometry'):
            trained = Path(raw['workspace'])/'runs'/raw['runs'][variant]
            validate_frozen_training(trained)
            for stage in ('train', 'predict_val'):
                assert read_json(trained/stage/'state.json')['status'] == 'completed'
        stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        p = derive_protocol(raw, previous, original, stamp)
        p['recovery'].update(original_protocol_sha256=sha256(original/'protocol.json'),
            recovery_script_sha256=sha256(__file__), created_at=dt.datetime.now(dt.timezone.utc).isoformat())
        output.mkdir(parents=True, exist_ok=False)
        for name in ('state.json', 'queue.json'):
            write_json(output/('interrupted_'+name), read_json(original/name))
        write_json(output/'protocol.json', p)
        write_json(output/'state.json', {'status': 'starting', 'completed': p['recovery']['completed']})
        interrupted = Path(raw['workspace'])/'runs'/raw['runs']['support']/'train'
        interrupted_state = read_json(interrupted/'state.json')
        interrupted_state.update(status='interrupted', detected_at=p['recovery']['created_at'],
            replacement_run=p['runs']['support'], selection_eligible=False)
        write_json(interrupted/'state.json', interrupted_state)
        previous.update(status='interrupted', detected_at=p['recovery']['created_at'], recovery_queue=str(output))
        write_json(original/'state.json', previous)
        old_tasks = read_json(original/'queue.json')
        old_tasks.update(status='interrupted', recovery_queue=str(output))
        for task in old_tasks['tasks']:
            if task['status'] == 'running':
                task['status'] = 'interrupted'
        write_json(original/'queue.json', old_tasks)
        env = os.environ.copy()
        env.update(PYTHONPATH=p['source'], PYTHONUNBUFFERED='1')
        env['PATH'] = str(Path(sys.executable).parent)+os.pathsep+env['PATH']
        with (output/'pipeline.log').open('w') as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                '--original', str(original), '--output', str(output), '--run'],
                cwd=p['source'], env=env, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
        write_json(output/'launcher.json', {'pid': child.pid})
        print(json.dumps({'output': str(output), 'pid': child.pid, 'support_run': p['runs']['support']}))
        return

    p = read_json(output/'protocol.json')
    assert sha256(original/'protocol.json') == p['recovery']['original_protocol_sha256']
    assert sha256(__file__) == p['recovery']['recovery_script_sha256']
    state = {'status': 'running', 'controller_pid': os.getpid(), 'completed': p['recovery']['completed'][:]}
    tasks = [{'name': name, 'status': 'completed' if name in state['completed'] else 'pending'} for name in p['order']]

    def save():
        state['updated_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(output/'state.json', state)
        write_json(output/'queue.json', {'status': state['status'], 'tasks': tasks})

    def begin(name):
        assert name == p['order'][len(state['completed'])], 'Unexpected task ordering'
        verify(p)
        state['stage'] = name
        next(t for t in tasks if t['name'] == name)['status'] = 'running'
        save()

    def done(name):
        next(t for t in tasks if t['name'] == name)['status'] = 'completed'
        state['completed'].append(name)
        state.pop('child_pid', None)
        save()

    def command(name, argv):
        begin(name)
        with (output/(name+'.log')).open('w') as log:
            child = subprocess.Popen([sys.executable, *argv], stdout=log, stderr=subprocess.STDOUT)
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
        trained = Path(p['workspace'])/'runs'/p['runs']['support']
        for action, label in (('train', 'train'), ('predict', 'source_predict')):
            command('support_'+label, ['-m', 'methods.support_reference', action,
                '--config', p['configs']['support'], '--run-id', p['runs']['support']])
        begin('support_formal_gate')
        validate_frozen_training(trained)
        assert read_json(trained/'train/initialization.json') == read_json(
            Path(p['workspace'])/'runs/maskdino-mssl-control-s42-20261018T025946/train/initialization.json')
        assert read_json(trained/'train/reference_initialization.json')['native_parameters_sha256'] == p['shared_native_initialization_sha256']
        assert read_json(trained/'train/reference_optimizer_contract.json')['status'] == 'passed'
        done('support_formal_gate')
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
        for variant in ('blend05', 'geometry', 'support'):
            trained = Path(p['workspace'])/'runs'/p['runs'][variant]
            _, checkpoint, digest = validate_frozen_training(trained)
            external = output/'external'/variant
            external.mkdir(parents=True, exist_ok=False)
            write_json(external/'images.json', {'images': gt['images']})
            write_json(external/'job.json', {'experiment': str(trained/'experiment.json'),
                'checkpoint': str(checkpoint), 'checkpoint_sha256': digest,
                'output': str(external), 'image_root': p['image_root'],
                'all_images': str(external/'images.json'),
                'source_predictions': str(trained/'predict_val/predictions.bbox.json')})
            command(variant+'_polypgen_inference', ['-m', 'methods.support_reference.external', '--job', str(external/'job.json')])
            begin(variant+'_polypgen_scoring')
            score_cohort(p['annotations'], read_json(external/'predictions.all_stills.bbox.json'), external/'primary')
            done(variant+'_polypgen_scoring')
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
    main()
