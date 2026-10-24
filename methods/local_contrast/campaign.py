"""Finite paired L1/L2 queue; cold smoke -> formal training -> external scoring."""
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

VARIANTS = ('box', 'mask_priority')
SMOKES = ('off', *VARIANTS)


def launch(output, acceptance):
    from methods.mssl.config import MSSLOptions, load_experiment
    from methods.support_reference.smoke_campaign import freeze_source, COMMON
    from methods.mssl.run import validate_dataset
    from polypbench.zeroshot import validate_frozen_training
    from .config import Options, validate
    root = Path.cwd().resolve()
    output, acceptance = output.resolve(), acceptance.resolve()
    gate = read_json(acceptance/'verification.json')
    assert gate['status'] == 'passed'
    if shutil.disk_usage(root).free < 15*2**30:
        raise RuntimeError('Need at least 15GiB free')
    base, _, _ = validate_frozen_training(root/'runs/mssl-official50-s42-20261018T025946')
    base = copy.deepcopy(base)
    base['mssl'] = asdict(MSSLOptions(**base['mssl']))
    base.pop('source_config', None)
    output.mkdir(parents=True, exist_ok=False)
    source = output/'source'
    freeze_source(root, source)
    configs, runs = {}, {}
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    for variant in SMOKES:
        experiment = copy.deepcopy(base)
        experiment.update(project_root='..', name='local_'+variant+'_s42',
            purpose='paired_mssl_local_contrast_L1_L2', local_contrast=asdict(Options(variant=variant)),
            local_expected_initialization_sha256=gate['initialization_sha256'])
        path = source/'configs'/f'{variant}.yaml'
        path.write_text(yaml.safe_dump(experiment, sort_keys=False), encoding='utf-8')
        resolved = validate(load_experiment(path))
        for key in (*COMMON, 'mssl'):
            assert resolved[key] == base[key], 'Common protocol changed: '+key
        configs[variant] = str(path)
        runs[variant] = 'local-'+variant+'-s42-'+stamp
    # Freeze source-domain data before training. Target metadata is frozen for
    # later scoring; only image-only records are ever passed into inference.
    data = validate_dataset(resolved)
    write_json(output/'source_data_manifest.json', data)
    parent = read_json(root/'runs/support-reference-recovery-20261022/protocol.json')
    refs = [acceptance/'verification.json', acceptance/'cases.json',
            root/'runs/polypgen-zeroshot-20261018/mssl/pooled/predictions.bbox.json']
    cohort_dir = Path(parent['annotations']).parent
    refs += [cohort_dir/f for f in ('provided_boxes.coco.json', 'image_manifest.json')]
    refs += [cohort_dir/f'C{c}.coco.json' for c in range(1, 7)]
    order = ['unit_checks', 'mssl_regression']+[v+'_cold_smoke' for v in SMOKES]
    order += ['common_smoke_gate']+[v+'_startup_precheck' for v in VARIANTS]
    order += [v+'_'+a for v in VARIANTS for a in ('train', 'source_predict', 'formal_gate')]
    order += ['target_image_hashes']+[v+'_'+a for v in VARIANTS for a in ('polypgen_inference', 'polypgen_scoring')]
    order += ['final_comparison']
    protocol = dict(workspace=str(root), source=str(source), configs=configs, runs=runs,
        variants=list(VARIANTS), smoke_variants=list(SMOKES), order=order,
        acceptance=str(acceptance), source_data_manifest=str(output/'source_data_manifest.json'),
        annotations=parent['annotations'], image_manifest=parent['image_manifest'], image_root=parent['image_root'],
        mssl_predictions=str(refs[2]), reference_metrics=parent['reference_metrics'],
        expected_initialization_sha256=gate['initialization_sha256'],
        reference_input_sha256={str(path): sha256(path) for path in refs},
        source_sha256={path.relative_to(source).as_posix(): sha256(path) for path in source.rglob('*')
                       if path.is_file() and 'third_party' not in path.relative_to(source).parts},
        budget='Seed42;2750 planned iterations;micro1 x accumulation16;original ImageNet R50 cold start',
        selection='max Kvasir Val bbox AP every275, earliest tie; PolypGen only after BOTH formal trains',
        treatment='L1 Box vs L2 Mask priority; shared paired eligibility/fallback/counts; direct normalized Encoder stride8',
        smoke_updates=20, no_smoke_warm_start=True, no_intermediate_ap_tuning=True,
        reserved=['PolypsSet', 'PolypDB'], copy_paste=False,
        created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        git_commit=subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip())
    write_json(output/'protocol.json', protocol)
    write_json(output/'state.json', dict(status='starting', completed=[]))
    env = os.environ.copy()
    env.update(PYTHONPATH=str(source)+os.pathsep+base['repo']['path'], PYTHONUNBUFFERED='1')
    env['PATH'] = str(Path(sys.executable).parent)+os.pathsep+env['PATH']
    with (output/'pipeline.log').open('w') as log:
        child = subprocess.Popen([sys.executable, '-m', 'methods.local_contrast.campaign', 'run',
            '--output', str(output)], cwd=source, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(output/'launcher.json', dict(pid=child.pid, order=order))
    print(dict(output=str(output), pid=child.pid, order=order))


def execute(output):
    p = read_json(output/'protocol.json')
    root = Path(p['workspace'])
    state = dict(status='running', controller_pid=os.getpid(), completed=[])
    tasks = [dict(name=n, status='pending') for n in p['order']]
    def save():
        state['updated_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(output/'state.json', state)
        write_json(output/'queue.json', dict(status=state['status'], tasks=tasks))
    def begin(name):
        assert p['order'][len(state['completed'])] == name
        for relative, digest in p['source_sha256'].items():
            assert sha256(Path(p['source'])/relative) == digest, 'Frozen code changed '+relative
        for path, digest in p['reference_input_sha256'].items():
            assert sha256(path) == digest, 'Reference input changed '+path
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
            raise RuntimeError(name+' failed; see stage log')
        done(name)
    try:
        save()
        command('unit_checks', ['-m', 'unittest', 'tests.test_local_contrast', '-v'])
        command('mssl_regression', ['-m', 'unittest', 'discover', '-s', 'methods/mssl/tests', '-v'])
        for variant in SMOKES:
            command(variant+'_cold_smoke', ['-m', 'methods.local_contrast.smoke', '--config', p['configs'][variant],
                '--output', str(output/'smoke'/variant), '--updates', str(p['smoke_updates'])])
        begin('common_smoke_gate')
        receipts = [read_json(output/'smoke'/v/'verification.json') for v in SMOKES]
        assert all(r['status'] == 'passed' and r['successful_updates'] == p['smoke_updates'] for r in receipts)
        assert all(r['initial_native_sha256'] == p['expected_initialization_sha256'] for r in receipts)
        sequences = [read_json(output/'smoke'/v/'data_sequence.json') for v in SMOKES]
        count = min(map(len, sequences))
        assert all(s[:count] == sequences[0][:count] for s in sequences[1:]), 'Augmentation prefix differs'
        for v in VARIANTS:
            rows = read_json(output/'smoke'/v/'updates.json')
            assert any(r.get('local_contrast/used', 0) > 0 for r in rows), 'No active local supervision in smoke'
        write_json(output/'smoke/verification.json', dict(status='passed', receipts=receipts,
            identical_augmentation_prefix_microbatches=count, no_formal_warm_start=True))
        done('common_smoke_gate')
        from methods.mssl.config import load_experiment
        from methods.mssl.run import validate_dataset
        for variant in VARIANTS:
            exp = load_experiment(p['configs'][variant])
            assert validate_dataset(exp) == read_json(p['source_data_manifest']), 'Source data changed'
            exp['local_startup_only'] = True
            startup = output/'startup'/variant
            startup.mkdir(parents=True, exist_ok=False)
            write_json(startup/'experiment.json', exp)
            command(variant+'_startup_precheck', ['-m', 'methods.mssl.worker', '--experiment',
                str(startup/'experiment.json'), '--output', str(startup), '--mode', 'train'])
            assert read_json(startup/'local_startup_gate.json')['status'] == 'passed'
        from polypbench.zeroshot import validate_frozen_training, score_cohort
        for variant in VARIANTS:
            for mode, label in (('train', 'train'), ('predict', 'source_predict')):
                command(variant+'_'+label, ['-m', 'methods.local_contrast', mode, '--config', p['configs'][variant],
                    '--run-id', p['runs'][variant]])
            begin(variant+'_formal_gate')
            trained = root/'runs'/p['runs'][variant]
            exp, _, _ = validate_frozen_training(trained)
            assert read_json(trained/'train/state.json')['status'] == 'completed'
            assert read_json(trained/'train/initialization.json') == read_json(
                root/'runs/mssl-official50-s42-20261018T025946/train/initialization.json')
            assert read_json(trained/'train/local_initialization.json')['native_sha256'] == p['expected_initialization_sha256']
            assert validate_dataset(exp) == read_json(p['source_data_manifest']), 'Source data changed'
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
            write_json(external/'images.json', dict(images=gt['images']))
            write_json(external/'job.json', dict(experiment=str(trained/'experiment.json'),
                checkpoint=str(checkpoint), checkpoint_sha256=digest, output=str(external),
                image_root=p['image_root'], all_images=str(external/'images.json'),
                source_predictions=str(trained/'predict_val/predictions.bbox.json')))
            command(variant+'_polypgen_inference', ['-m', 'polypbench.adapters.zeroshot_worker',
                '--job', str(external/'job.json')])
            begin(variant+'_polypgen_scoring')
            assert sha256(checkpoint) == digest
            score_cohort(p['annotations'], read_json(external/'predictions.all_stills.bbox.json'), external/'primary')
            done(variant+'_polypgen_scoring')
        command('final_comparison', ['-m', 'methods.local_contrast.comparison', '--output', str(output)])
        assert state['completed'] == p['order']
        state['status'] = 'completed'
    except BaseException as error:
        state.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        for t in tasks:
            if t['status'] == 'running':
                t['status'] = 'failed'
        raise
    finally:
        save()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('launch', 'run'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--acceptance', type=Path)
    args = p.parse_args()
    if args.action == 'launch':
        if args.acceptance is None:
            p.error('--acceptance is required')
        launch(args.output, args.acceptance)
    else:
        execute(args.output.resolve())
