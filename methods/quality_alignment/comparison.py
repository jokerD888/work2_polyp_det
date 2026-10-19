"""Four-group descriptive contrasts with fixed, paired image resamples."""
import argparse
import json
from pathlib import Path
import numpy as np
from polypbench.io import read_json, write_json, sha256
from polypbench.analysis.coco_cache import CocoCache, metrics
from polypbench.analysis.uncertainty import run as bootstrap, interval


CONTRASTS={'C-A':{'baseline_q':1,'baseline':-1},'D-B':{'mssl_q':1,'mssl':-1},
           'D-C':{'mssl_q':1,'baseline_q':-1},'I=(D-B)-(C-A)':{'mssl_q':1,'mssl':-1,'baseline_q':-1,'baseline':1}}


def run(output):
    out=Path(output);p=read_json(out/'protocol.json');workspace=Path(p['workspace']);old=Path(p['previous_analysis'])
    previous=read_json(old/'protocol.json');gt=read_json(p['annotations'])
    predictions={label:read_json(previous['inputs']['predictions'][label]) for label in ('baseline','mssl')}
    for label in ('baseline_q','mssl_q'): predictions[label]=read_json(out/'external'/label/'pooled/predictions.bbox.json')
    destination=out/'four_group_comparison';destination.mkdir(exist_ok=False)
    uncertainty=bootstrap(gt,predictions,destination/'uncertainty',draws_path=old/'uncertainty/image_multiplicities.npy')
    samples=np.load(destination/'uncertainty/bootstrap_ap_vectors.npz');old_samples=np.load(old/'uncertainty/bootstrap_ap_vectors.npz')
    for key in samples.files:
        if key.startswith(('baseline__','mssl__')):np.testing.assert_allclose(samples[key],old_samples[key],rtol=0,atol=1e-12)
    assert sha256(destination/'uncertainty/image_multiplicities.npy')==sha256(old/'uncertainty/image_multiplicities.npy')
    source={};target={};training={}
    for label,run_id in p['runs'].items():
        trained=workspace/'runs'/run_id;e=read_json(trained/'experiment.json')
        vector=CocoCache(read_json(e['dataset']['val']['annotations']),read_json(trained/'predict_val/predictions.bbox.json')).reference
        source[label]={**metrics(vector),'AP50':float(vector[0])}
        vector=CocoCache(gt,predictions[label]).reference;target[label]={**metrics(vector),'AP50':float(vector[0])}
        rows=[json.loads(s) for s in (trained/'train/metrics.json').read_text().splitlines() if s.strip()]
        rows=[r for r in rows if 'optimizer/successful_updates' in r];last=max(rows,key=lambda r:r['iteration'])
        training[label]={'last_iteration':last['iteration'],'successful_updates':last['optimizer/successful_updates'],
            'last_amp_scale':last['amp/loss_scale'],'selected_checkpoint':read_json(trained/'train/selected_checkpoint.json')}
        if label in ('baseline_q','mssl_q'):
            probes=[json.loads(s) for s in (trained/'train/quality_gradient_probes.jsonl').read_text().splitlines()]
            training[label]['gradient_probes']=probes
            training[label]['last_logged_quality_stats']={k:v for k,v in last.items() if k.startswith('quality/')}
    point_contrasts={name:{split:{key:sum(w*values[label][key]*100 for label,w in weights.items())
        for key in ('AP','AP50','AP75','AP90','AP95')} for split,values in (('source',source),('polypgen',target))}
        for name,weights in CONTRASTS.items()}
    paired={}
    for name,weights in CONTRASTS.items():
        vector=sum(w*samples[label+'__pooled'] for label,w in weights.items())
        selected={'AP':vector.mean(1),'AP75':vector[:,5],'AP90':vector[:,8],'AP95':vector[:,9]}
        paired[name]={key:{'point_delta_pp':point_contrasts[name]['polypgen'][key],**interval(value*100)}
                      for key,value in selected.items()}
    summary={'source_fraction':source,'polypgen_fraction':target,'contrasts_pp':point_contrasts,
        'paired_polypgen_intervals':paired,'training':training,'original_A_B_bootstrap_parity':'passed',
        'source_diagnosis':str(out/'source_quality_diagnostics/diagnostics.json'),
        'attribution_limit':p['attribution_limit'],'uncertainty_limits':uncertainty['protocol']['limitations'],
        'decision_scope':'Fixed one-seed exploratory component experiment. No automatic new module or additional training.'}
    write_json(destination/'comparison.json',summary)
    lines=['# IA-BCE-Final：固定 2×2 对照','','A=Baseline；B=MSSL V1；C=Baseline+Q；D=MSSL+Q。',
        '两组新模型均从共同ImageNet初始化完整训练；不从Smoke或A/B接续。源域Val参与选权，PolypGen为已观察的外部开发复核。','']
    for name,values in (('Kvasir Val',source),('PolypGen固定1473张',target)):
        lines += [f'## {name}','','| 模型 | AP | AP50 | AP75 | AP90 | AP95 |','|---|---:|---:|---:|---:|---:|']
        for label,v in values.items():lines.append('| '+label+' | '+' | '.join(f'{100*v[k]:.3f}' for k in ('AP','AP50','AP75','AP90','AP95'))+' |')
        lines.append('')
    lines+=['## 预定增量（AP百分点）','','| 对比 | Kvasir Val | PolypGen | PolypGen图像级95%区间 |', '|---|---:|---:|---|']
    for name,value in point_contrasts.items():
        ci=paired[name]['AP'];lines.append(f"| {name} | {value['source']['AP']:+.3f} | {value['polypgen']['AP']:+.3f} | [{ci['low_pp']:+.3f}, {ci['high_pp']:+.3f}] |")
    lines+=['','I只作描述性交互，单Seed与图像重采样不能证明训练交互稳定性。原2000次按中心配对抽样逐字节复用，A/B全部历史向量复核通过。','',
        '## 读数与研究边界','',
        '同时检查C−A、D−B与D−C；不只凭D>B宣布组合有效。若只在源域改善而外部回退，不直接采用组合。',
        '若C、D均有收益但D−C接近零，成熟Q可能解释主要收益；不能强称Mask特定协同。若收益不清或冲突，允许保留V1。',
        '本轮同时改变软目标、正负项权重与Focal损失形状，收益不能全部归因为质量排序。相同外部系数4不表示梯度强度配平。',
        'IA-BCE是固定作者组件移植，不自动构成第二创新点。是否需要Mask特定机制须结合源域候选覆盖、组内错排、重复检测及下列梯度记录人工判断。','',
        '## 工程记录','', '| 模型 | 最后迭代 | 成功更新 | AMP Scale |','|---|---:|---:|---:|']
    for label,t in training.items():lines.append(f"| {label} | {int(t['last_iteration'])} | {int(t['successful_updates'])} | {t['last_amp_scale']} |")
    lines+=['','comparison.json保留各组选权记录、Q组稀疏分类头梯度探针、软目标分布与正负损失。',
        '每组初始/最佳Checkpoint验收在 ../*_precheck/report.json、../*_checkpoint_replay/report.json。',
        '同GT错排、全300与Top100覆盖、官方重复FP及分类/Mask融合诊断在 ../source_quality_diagnostics/。',
        'PolypsSet、PolypDB未使用；没有按中途AP调整配置或自动追加训练。']
    report='\n'.join(lines)+'\n';(destination/'summary.md').write_text(report,encoding='utf-8');(out/'summary.md').write_text(report,encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);run(parser.parse_args().output)
