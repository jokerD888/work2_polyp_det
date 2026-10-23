"""Predeclared AP/geometry/net-coverage/FP comparison after all full runs."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from polypbench.io import read_json, write_json
from polypbench.analysis.matching import xyxy
from polypbench.analysis.candidate_stage import stage_metrics
from polypbench.analysis.initialization_intervention import signed_events


def run(output):
    from polypbench.analysis.coco_cache import official_eval, ap_vector
    from polypbench.analysis.quality_diagnostic import fp_types
    p = read_json(output/'protocol.json')
    gt = read_json(p['annotations'])
    image_areas = {im['id']: im['height']*im['width'] for im in gt['images']}
    area_cuts = np.quantile([a['bbox'][2]*a['bbox'][3] / image_areas[a['image_id']]
        for a in gt['annotations']], [.25, .5, .75]).tolist()
    anns = defaultdict(list)
    for a in gt['annotations']:
        anns[a['image_id']].append(a)
    reference = Path(p['native_stages'])
    rows = {}
    for variant in ('mask_extra', 'encoder_extra'):
        external = output/'external'/variant
        predictions = read_json(external/'predictions.all_stills.bbox.json')
        ev = official_eval(gt, predictions)
        vector = ap_vector(ev)
        metrics = {'AP': float(vector.mean()), 'AP50': float(vector[0]),
                   'AP75': float(vector[5]), 'AP90': float(vector[8]), 'AP95': float(vector[9])}
        groups = defaultdict(list)
        stage_totals = defaultdict(lambda: defaultdict(int))
        for image in gt['images']:
            truth = xyxy([a['bbox'] for a in anns[image['id']]])
            with np.load(external/'candidates'/f"{image['id']}.npz") as arrays:
                stages = {}
                for stage in ('encoder', 'mask_native', 'initial', 'final300'):
                    _, s = stage_metrics(arrays[stage+'_boxes_xyxy'], truth)
                    stages[stage] = s
                    for threshold, vals in s['thresholds'].items():
                        stage_totals[stage]['one_to_one_'+threshold] += vals['one_to_one_coverage']
                        stage_totals[stage]['marginal_'+threshold] += vals['marginal_gt_covered']
                _, top100 = stage_metrics(arrays['top100_boxes_xyxy'], truth)
            old = read_json(reference/'images'/f"{image['id']}.json")
            assert len(old['stages']['decoder_09']['max_iou']) == len(truth)
            for j, ann in enumerate(anns[image['id']]):
                # Raw legacy report has its own annotation order; join by ID.
                idx = old['annotation_ids'].index(ann['id'])
                row = {'image_id': image['id'], 'annotation_id': ann['id'],
                    'native': old['stages']['decoder_09']['max_iou'][idx],
                    'new': stages['final300']['max_iou'][j],
                    'top100': top100['max_iou'][j],
                    'area_ratio': ann['bbox'][2]*ann['bbox'][3]/(image['height']*image['width'])}
                quartile = 1 + int(np.searchsorted(area_cuts, row['area_ratio'], side='left'))
                for label in ('all', 'single' if len(truth) == 1 else 'multi', 'center='+old['center'],
                              'relative_area_q'+str(quartile)):
                    groups[label].append(row)
        events = {label: {str(t): signed_events([r['native'] for r in values],
                    [r['new'] for r in values], t) for t in (.5, .75, .9)} for label, values in groups.items()}
        old_metrics = p['reference_metrics']['baseline']
        record = {'bbox': metrics, 'delta_AP_to_native': metrics['AP']-old_metrics['AP'],
                  'stages': dict(stage_totals), 'marginal_gt_events': events,
                  'fp_types': fp_types(ev, gt), 'gt_rows': groups['all'],
                  'negative_frames': read_json(external/'primary/negative_frame_diagnostic.json'),
                  'event_semantics': 'GT pool maxIoU across separately trained models; not query-identity recovery'}
        write_json(external/'detailed_comparison.json', record)
        rows[variant] = record
    compact = {'units': 'fraction_0_to_1', 'reference_metrics': p['reference_metrics'],
        'gt_relative_area_quartile_cuts': area_cuts,
        'conditions': {v: {k: x[k] for k in ('bbox', 'delta_AP_to_native', 'stages', 'marginal_gt_events', 'fp_types', 'negative_frames')}
                       for v, x in rows.items()},
        'reference_diagnostics': p.get('reference_diagnostics'),
        'selection': 'Kvasir Val checkpoint only; PolypGen external development assessment',
        'no_automatic_mssl_combination': True, 'no_innovation_claim_from_single_seed': True}
    write_json(output/'comparison.json', compact)
    lines = ['# 第一层双参照特征读取独立对照完成', '', '| 条件 | PolypGen AP | AP50 | AP75 | AP90 | 相对原生AP |',
             '|---|---:|---:|---:|---:|---:|']
    for variant, r in rows.items():
        m = r['bbox']
        lines.append(f"| {variant} | {100*m['AP']:.3f} | {100*m['AP50']:.3f} | {100*m['AP75']:.3f} | "
                     f"{100*m['AP90']:.3f} | {100*r['delta_AP_to_native']:+.3f} |")
    lines += ['', '完整coverage净得失、各中心/单多目标、重复FP及定位FP见comparison.json和各组detailed_comparison.json。',
        '固定同预算单seed实验，不依中途AP调参；不自动启动MSSL组合，不把工程头或单seed增益直接称为第二创新。']
    (output/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(Path(parser.parse_args().output))
