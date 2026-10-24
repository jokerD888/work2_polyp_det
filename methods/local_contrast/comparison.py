"""AP, matched recall/misses and FP at fixed maxDets and group definitions."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from polypbench.io import read_json, write_json
from polypbench.analysis.coco_cache import official_eval, ap_vector
from polypbench.analysis.quality_diagnostic import fp_types
from polypbench.analysis.matching import xyxy, iou_matrix


def metrics(gt, predictions):
    ev = official_eval(gt, predictions)
    vector = ap_vector(ev)
    ap = dict(AP=float(vector.mean()), AP50=float(vector[0]), AP75=float(vector[5]),
              AP90=float(vector[8]), AP95=float(vector[9]))
    fp = fp_types(ev, gt)
    count = sum(not a.get('iscrowd', 0) for a in gt['annotations'])
    recall = {threshold: dict(gt=count, matched=values['100']['tp'],
        misses=count-values['100']['tp'], recall=values['100']['tp']/count if count else None)
        for threshold, values in fp.items()}
    return dict(bbox=ap, fp_types=fp, matched_recall_at_100=recall)


def run(output):
    p = read_json(output/'protocol.json')
    gt = read_json(p['annotations'])
    anns = defaultdict(list)
    for a in gt['annotations']:
        anns[a['image_id']].append(a)
    groups = defaultdict(set)
    per_image = {im['id']: im for im in gt['images']}
    for image in gt['images']:
        i = image['id']
        groups['all'].add(i)
        groups['negative' if not anns[i] else ('single' if len(anns[i]) == 1 else 'multi')].add(i)
        # The exact center source is the preexisting cohort's C1...C6 files.
    for c in range(1, 7):
        cohort = Path(p['annotations']).parent/f'C{c}.coco.json'
        groups[f'C{c}'] = {im['id'] for im in read_json(cohort)['images']}
    areas = [a['bbox'][2]*a['bbox'][3]/(per_image[a['image_id']]['height']*per_image[a['image_id']]['width']) for a in gt['annotations']]
    cuts = np.quantile(areas, [.25, .5, .75]).tolist()
    old = Path(p['mssl_predictions'])
    conditions = {'mssl': read_json(old)}
    for variant in ('box', 'mask_priority'):
        conditions[variant] = read_json(output/'external'/variant/'predictions.all_stills.bbox.json')
    results = {}
    for variant, predictions in conditions.items():
        pred_by = defaultdict(list)
        for row in predictions:
            pred_by[row['image_id']].append(row)
        record = metrics(gt, predictions)
        grouped = {}
        for name, ids in groups.items():
            if name == 'negative':
                continue
            sub = dict(gt, images=[im for im in gt['images'] if im['id'] in ids],
                       annotations=[a for a in gt['annotations'] if a['image_id'] in ids])
            grouped[name] = metrics(sub, [r for r in predictions if r['image_id'] in ids])
        gt_rows = []
        for image in gt['images']:
            truths = anns[image['id']]
            pool = sorted(pred_by[image['id']], key=lambda r: -r['score'])[:100]
            matrix = iou_matrix(xyxy([a['bbox'] for a in truths]), xyxy([r['bbox'] for r in pool]))
            best = matrix.max(1) if len(pool) else np.zeros(len(truths))
            for a, value in zip(truths, best):
                area = a['bbox'][2]*a['bbox'][3]/(image['height']*image['width'])
                gt_rows.append(dict(image_id=image['id'], annotation_id=a['id'], max_iou=float(value),
                    relative_area=area, area_quartile=1+int(np.searchsorted(cuts, area, side='left'))))
        record['groups'] = grouped
        record['area_quartile_coverage'] = {str(q): {str(t): dict(
            total=sum(r['area_quartile'] == q for r in gt_rows),
            misses=sum(r['area_quartile'] == q and r['max_iou'] < t for r in gt_rows))
            for t in (.5, .75, .9, .95)} for q in range(1, 5)}
        record['gt_rows'] = gt_rows
        negatives = groups['negative']
        record['negative_frames_at_score05'] = dict(images=len(negatives),
            frames_with_fp=sum(any(r['score'] >= .5 for r in pred_by[i]) for i in negatives),
            fp=sum(r['score'] >= .5 for i in negatives for r in pred_by[i]))
        results[variant] = record
    deltas = {v: {ref: {key: 100*(results[v]['bbox'][key]-results[ref]['bbox'][key])
        for key in results[v]['bbox']} for ref in ('mssl', 'box') if ref != v}
        for v in ('box', 'mask_priority')}
    write_json(output/'comparison.json', dict(units='AP fractions; delta percentage points',
        conditions=results, deltas=deltas, area_quartile_cuts=cuts,
        coverage_semantics='Top100 GT maxIoU; FP and matched recall in official COCO fp_types',
        decision='Primary AP50:95; jointly inspect strict IoU and detection tradeoffs; no automatic adoption from single seed',
        selection='Kvasir Val AP only; PolypGen is an observed external development cohort'))
    lines = ['# L1/L2 固定对照结果', '', '| 条件 | AP | AP50 | AP75 | AP90 | AP95 |', '|---|---:|---:|---:|---:|---:|']
    for v, r in results.items():
        lines.append('| '+v+' | '+' | '.join(f'{100*r["bbox"][key]:.3f}' for key in ('AP', 'AP50', 'AP75', 'AP90', 'AP95'))+' |')
    lines += ['', '分中心、单/多目标、相对面积分组、漏检与误报见 comparison.json。',
              '单 Seed 外部开发结果；积极结果后再做配对多 Seed 与必要消融。']
    (output/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args().output.resolve())
