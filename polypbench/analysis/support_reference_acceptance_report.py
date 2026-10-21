"""CPU-only summary of completed engineering evidence; no model imports."""
import argparse
from collections import defaultdict
from pathlib import Path
import statistics

from polypbench.io import read_json, write_json, sha256


def describe(values):
    return {'n': len(values), 'minimum': min(values) if values else None,
            'median': statistics.median(values) if values else None,
            'maximum': max(values) if values else None}


def run(directory):
    verification = read_json(directory/'verification.json')
    assert verification['status'] == 'passed'
    result = read_json(directory/'results.json')
    groups = defaultdict(list)
    for row in result['train']:
        groups[row['model']].append(row)
    summary = {'decision': 'GO_CONTROLLED_TRAINING_DESIGN_NOT_PERFORMANCE_VALIDATION',
        'optimizer_steps': 0, 'verification': verification,
        'input_results_sha256': sha256(directory/'results.json'), 'models': {}}
    for model, rows in groups.items():
        matches = [m for r in rows for m in r['supervision']['matches']]
        strata = {}
        for name in sorted({r['stratum'] for r in rows}):
            items = [r for r in rows if r['stratum'] == name]
            ms = [m for r in items for m in r['supervision']['matches']]
            strata[name] = {'views': len(items), 'matches': len(ms),
                            'eligible_matches': sum(m['adapter_eligible'] for m in ms)}
        gradients = {}
        for name in sorted({k for r in rows for k in r['gradients']}):
            items = [r['gradients'][name] for r in rows if name in r['gradients']]
            gradients[name] = {'adapter_norm': describe([g['adapter_l2'] for g in items]),
                'positive_adapter_norm_views': sum(g['adapter_l2'] > 0 for g in items),
                'all_finite': all(g['finite'] for g in items),
                'native_box_bias_norm': describe([g['native_bbox_last_bias_l2'] for g in items]),
                'native_mask_bias_norm': describe([g['native_mask_last_bias_l2'] for g in items])}
        times = result['timing'][model]
        off = statistics.median(times['seconds']['False'])
        on = statistics.median(times['seconds']['True'])
        summary['models'][model] = {
            'unique_images': len({r['image_id'] for r in rows}), 'augmented_views': len(rows),
            'query_observations': sum(r['supervision']['all_queries'] for r in rows),
            'eligible_query_observations': sum(r['supervision']['all_query_eligible'] for r in rows),
            'gt_matches': len(matches), 'eligible_gt_matches': sum(m['adapter_eligible'] for m in matches),
            'empty_gt_mask_matches': sum(m['gt_mask_pixels'] == 0 for m in matches),
            'views_without_retained_gt': sum(r['supervision']['post_lsj_gt'] == 0 for r in rows),
            'views_without_eligible_matches': sum(not any(m['adapter_eligible'] for m in r['supervision']['matches']) for r in rows),
            'initial_matched_box_iou': describe([m['initial_box_iou'] for m in matches]),
            'strata': strata, 'gradients': gradients,
            'peak_allocated_GiB': max(r['peak_allocated_bytes'] for r in rows) / 2**30,
            'engineering_timing': {'disabled_median_seconds': off, 'enabled_median_seconds': on,
                'difference_seconds': on-off, 'relative_difference': on/off-1,
                'scope': times['scope'], 'five_repeats_not_dataset_benchmark': True}}
    write_json(directory/'engineering_summary.json', summary)
    lines = ['# 真实Detector工程验收完成', '',
        '**结论：工程路径通过，可以准备受控训练；尚无新方法AP或创新性证据。**', '',
        '仅Kvasir Train；已有成熟Baseline/MSSL权重；无optimizer、0次更新。'
        '同图两种增强不是独立样本。', '',
        '| 模型 | 图像/增强视图 | 有效匹配/全部GT匹配 | 有效Query观测/全部 | 峰值GiB |',
        '|---|---:|---:|---:|---:|']
    for model, values in summary['models'].items():
        lines.append(f"| {model} | {values['unique_images']}/{values['augmented_views']} | "
                     f"{values['eligible_gt_matches']}/{values['gt_matches']} | "
                     f"{values['eligible_query_observations']}/{values['query_observations']} | "
                     f"{values['peak_allocated_GiB']:.3f} |")
    lines += ['', f"FP32推理原生/关闭/零输出逐张一致：{len(result['eval'])}例；"
              f"AMP训练模式严格一致：{verification['train_cases']}视图；"
              f"真实batch2检查：{len(result['batch2'])}组；固定非零参考扰动：{len(result['nonzero'])}组。", '',
              '## 梯度路由', '', '| 模型 | 损失组 | adapter范数中位数 | 非零视图/总数 |',
              '|---|---|---:|---:|']
    for model, values in summary['models'].items():
        for label, gradients in values['gradients'].items():
            g = gradients['adapter_norm']
            lines.append(f"| {model} | {label} | {g['median']:.6g} | "
                         f"{gradients['positive_adapter_norm_views']}/{g['n']} |")
    lines += ['', '所有分解梯度有限；aux0框损失对被测原生Box/Mask末层bias的新增路径为0；'
              'Encoder interm与DN初始损失对adapter为0。后续DN可经普通Query产生间接梯度。'
              '范数来自固定64倍诊断反传后在FP32除回，不是参数更新，也不是梯度越大越有效。', '',
              '## 工程时耗', '', '| 模型 | 关闭/开启中位数秒 | 差值毫秒 |', '|---|---:|---:|']
    for model, values in summary['models'].items():
        t = values['engineering_timing']
        lines.append(f"| {model} | {t['disabled_median_seconds']:.4f}/{t['enabled_median_seconds']:.4f} | "
                     f"{t['difference_seconds']*1000:.2f} |")
    lines += ['', '每模型仅1张图、5次预热后重复、FP32、关闭观察器。'
              '这不是全数据集FPS基准，不能据此声称零开销或稳定吞吐提升。', '',
              '## 边界与下一步', '',
              '工程证据不回答从原始预训练开始是否收敛，也未验证空间profile的独立价值。'
              '下一阶段须按同起点/预算比较普通几何校准、空间支持校准和固定融合重训，'
              '先验证Baseline独立作用，再决定MSSL组合。PolypGen仍为主要外部开发评价，'
              'PolypsSet/PolypDB保留；本轮没有读取它们。']
    (directory/'engineering_summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True)
    run(Path(parser.parse_args().directory))
