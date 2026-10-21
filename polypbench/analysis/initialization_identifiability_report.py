"""Render completed CPU audit; no model, classifier, gate or cutoff tuning."""
import argparse
from pathlib import Path
import numpy as np
from polypbench.io import read_json,write_json,sha256
from .initialization_identifiability import FEATURES,CLASSES,outcome,balanced_weights,rank_metrics,subset_report,weighted_quantiles


def number(x):
    return '缺失/不可估计' if x is None else f'{x:.3f}'


def useful_witnesses(native_query_iou, alternative_query_iou, owners, native_gt_max, alternative_gt_max, threshold):
    """GT-dependent outcome labels, not feature extraction or deployable selection.

    Improving a redundant query is NOT the same as restoring GT-pool coverage.
    """
    unique=owners>=0
    witness=np.zeros(len(owners),bool)
    witness[unique]=(native_gt_max[owners[unique]]<threshold)&(alternative_gt_max[owners[unique]]>=threshold)
    return witness&(native_query_iou<threshold)&(alternative_query_iou>=threshold)


def gt_query_correspondence(out,key):
    records=read_json(out/(key+'.per_gt.json'))
    lookup={}
    for condition in ('encoder','blend05'):
        by_image={}
        for r in records:
            if r['condition']==condition:
                by_image.setdefault(r['image_id'],[]).append(r)
        lookup[condition]=by_image
    with np.load(out/(key+'.query_audit.npz')) as a:
        features=a['features']; query_ids=a['query_id']; images=a['image_index']; ids=a['image_ids']
        owner=a['owner_gt_index']; native=a['native_final_max_any_gt_iou']
        assert len(owner)==300*len(ids)
        np.testing.assert_array_equal(images.reshape(-1,300)[:,0],np.arange(len(ids)))
        result={}
        for condition in ('encoder','blend05'):
            alternative=a[condition+'_final_max_any_gt_iou']; by_threshold={}
            for threshold in (.5,.75,.9):
                witness=np.zeros(len(owner),bool)
                for index,iid in enumerate(ids):
                    rows=lookup[condition].get(int(iid),[])
                    n=np.array([r['stages']['decoder_09']['native'] for r in rows])
                    b=np.array([r['stages']['decoder_09']['alternative'] for r in rows])
                    pick=slice(index*300,(index+1)*300)
                    witness[pick]=useful_witnesses(native[pick],alternative[pick],owner[pick],n,b,threshold)
                condition_result={}
                for name,selection in {'all300':np.ones(len(owner),bool),'early_top10':query_ids<10,'early_top1':query_ids==0}.items():
                    w=balanced_weights(images,selection); y=witness[selection]
                    recovered=(native[selection]<threshold)&(alternative[selection]>=threshold)
                    condition_result[name]={'all_recovered_queries':int(recovered.sum()),
                        'unique_gt_actual_recovery_witness_queries':int(y.sum()),
                        'actual_recovery_witness_images':int(len(np.unique(images[selection][y]))),
                        'other_query_recovery_not_verified_as_new_GT':int((recovered&~y).sum()),
                        'features':{f:rank_metrics(features[selection,j],y,w) for j,f in enumerate(FEATURES)}}
                    if threshold==.5:
                        # Correct joint objective: recover LOW-IoU misses while
                        # protecting ORIGINAL high precision, not recovering .9 misses.
                        high=native[selection]>=.9
                        statistics={}
                        for j,f in enumerate(FEATURES):
                            v=features[selection,j]; valid=np.isfinite(v)
                            edges=np.unique(weighted_quantiles(v,w,np.linspace(0,1,11))) if valid.any() else np.array([])
                            def overlap(a,b):
                                if len(edges)<2 or not (a&valid).any() or not (b&valid).any(): return None
                                ha=np.histogram(v[a&valid],bins=edges,weights=w[a&valid])[0]
                                hb=np.histogram(v[b&valid],bins=edges,weights=w[b&valid])[0]
                                return float(np.minimum(ha/ha.sum(),hb/hb.sum()).sum())
                            pair=recovered|high; witness_pair=y|high
                            statistics[f]={
                                'coarse_recovery_vs_native_high_precision':rank_metrics(v[pair],recovered[pair],w[pair]),
                                'actual_GT_witness_vs_native_high_precision':rank_metrics(v[witness_pair],y[witness_pair],w[witness_pair]),
                                'coarse_recovery_high_precision_overlap':overlap(recovered,high),
                                'actual_GT_witness_high_precision_overlap':overlap(y,high),
                                'native_high_precision_quantiles_10_50_90':weighted_quantiles(v[high],w[high],[.1,.5,.9])}
                        condition_result[name]['cross_threshold_protection']={
                            'native_ge_09_queries':int(high.sum()),
                            'native_ge_09_images':int(len(np.unique(images[selection][high]))),
                            'features':statistics,
                            'limits':'Contrast .5 recovery against original .9 protection; not an inferred gate or new outcome threshold.'}
                by_threshold[str(threshold)]=condition_result
            result[condition]=by_threshold
    write_json(out/(key+'.gt_query_correspondence.json'),result)
    return result


def pre_gt_guard_subsets(out,key):
    """Confound controls selected ONLY by native pre-decoder validity fields."""
    with np.load(out/(key+'.query_audit.npz')) as a:
        x=a['features']; images=a['image_index']; native=a['native_final_max_any_gt_iou']
        subsets={'encoder_interior':x[:,FEATURES.index('encoder_boundary')]==0}
        legality=x[:,FEATURES.index('encoder_not_native_legal')]
        if np.isfinite(legality).any(): subsets['native_legal_encoder']=legality==0
        result={}
        for c in ('encoder','blend05'):
            result[c]={}
            for t in (.5,.75,.9):
                labels=outcome(native,a[c+'_final_max_any_gt_iou'],t)
                result[c][str(t)]={name:subset_report(x,labels,images,mask) for name,mask in subsets.items()}
    write_json(out/(key+'.pre_gt_guard_subsets.json'),result)
    return result


def run(output):
    out=Path(output)
    assert read_json(out/'state.json')['status']=='completed'
    assert read_json(out/'verification.json')['status']=='passed'
    inputs={}
    models={}
    lines=['# Box–Mask 初始化可识别性：CPU审计','',
        '只读取已完成缓存，未调用GPU、未加载模型、未拟合门控或阈值。PolypGen为主要开发基准，Kvasir Val为辅助复核；各模型的Query编号仅在模型内部追踪。','',
        '全部300 Query保留；early_top10/early_top1来自原生Encoder TopK的原始顺序，在初始化前确定，不用GT挑Query。GT只用于事后几何事件和归属标记。',
        '每个统计子集内每张图总权重相等，并同时报告原始Query/图像基数。排序AP是特征排序的平均精确率，不是bbox AP，也不是已训练门控的性能。','']
    for cohort in ('polypgen','kvasir_val'):
        for model in ('baseline','mssl'):
            key=cohort+'_'+model; path=out/(key+'.json'); inputs[str(path)]=sha256(path)
            for suffix in ('.per_gt.json','.query_audit.npz'):
                p=out/(key+suffix); inputs[str(p)]=sha256(p)
            value=read_json(path); models[key]=value
            witnesses=gt_query_correspondence(out,key)
            guards=pre_gt_guard_subsets(out,key)
            lines += [f'## {key}','',f"{value['images']}张、{value['gt']} GT、{value['queries']} Query。归属统计：`{value['ownership']}`。",'']
            lines += ['### 不按GT选样的前10 Query：类别基数','',
                '| 干预 | IoU阈值 | 两者失败 | 恢复 | 损坏 | 两者覆盖 | 恢复图像均衡占比 |',
                '|---|---:|---:|---:|---:|---:|---:|']
            for c in ('encoder','blend05'):
                for t in ('.5','.75','.9'):
                    t=str(float(t)); group=value['conditions'][c]['thresholds'][t]['early_top10']
                    counts=group['counts']; fraction=group['image_balanced_class_fraction']['recovered']
                    lines.append(f"| {c} | {t} | "+' | '.join(str(counts[k]) for k in CLASSES)+f' | {100*fraction:.3f}% |')
            lines += ['', '### 前10 Query、IoU≥0.5：预先固定的四种空间特征','',
                '| 干预/特征（高值预测恢复） | 恢复vs损坏 AUC | 恢复vs原生覆盖 AUC | 恢复vs全部 排序AP | 恢复基数 | 与两者覆盖分布重叠 |',
                '|---|---:|---:|---:|---:|---:|']
            for c in ('encoder','blend05'):
                feature=value['conditions'][c]['thresholds']['0.5']['early_top10']['features']
                for name in FEATURES[:4]:
                    f=feature[name]; m=f['metrics']
                    nums=[m['recovery_vs_damage']['auc'],m['recovery_vs_native_covered']['auc'],
                        m['recovery_vs_all_other']['ranking_ap'],m['recovery_vs_all_other']['prevalence'],
                        f['distribution_overlap']['recovery_both_good']]
                    lines.append('| '+c+'/'+name+' | '+' | '.join(number(n) for n in nums)+' |')
            lines += ['', '### 前10 Query：全特征结果（不挑选最好的一项）','',
                '| 干预/特征 | .5 恢复vs全部排序AP | .5 基数 | .9 恢复vs全部排序AP | .9 基数 |',
                '|---|---:|---:|---:|---:|']
            for c in ('encoder','blend05'):
                for name in FEATURES:
                    nums=[]
                    for t in ('0.5','0.9'):
                        m=value['conditions'][c]['thresholds'][t]['early_top10']['features'][name]['metrics']['recovery_vs_all_other']
                        nums.extend([m['ranking_ap'],m['prevalence']])
                    lines.append('| '+c+'/'+name+' | '+' | '.join(number(n) for n in nums)+' |')
            lines += ['', '### 预先固定空间特征的中心/尺寸/单多目标检查','',
                '| 干预/子组（前10 Query） | 恢复Query数 | 图像数 | 面积比排序AP/基数 | 中心偏移排序AP/基数 |',
                '|---|---:|---:|---:|---:|']
            for c in ('encoder','blend05'):
                for name,groups in value['conditions'][c]['strata'].items():
                    g=groups['0.5']
                    if not g['queries']:
                        lines.append(f'| {c}/{name} | 0 | 0 | 不可估计 | 不可估计 |'); continue
                    values=[]
                    for feature in FEATURES[1:3]:
                        m=g['features'][feature]['metrics']['recovery_vs_all_other']
                        values.append(number(m['ranking_ap'])+'/'+number(m['prevalence']))
                    lines.append(f"| {c}/{name} | {g['counts']['recovered']} | {g['images']} | "+' | '.join(values)+' |')
            lines += ['', '全部300、单个前1、非边界Encoder及原生合法Encoder子集，所有类别的分位数/缺失量/图像数/重叠度，以及.75/.9完整特征统计，见同名JSON。GT尺寸和唯一归属子集是事后诊断，不是免费推理选择规则。','']
            lines += ['### Query恢复是否真正见证GT池新增覆盖','',
                '| 干预，IoU≥0.5，前10 Query | Query恢复总数 | 唯一归属且GT池新增覆盖的见证Query | 见证图像 | 面积比见证排序AP/基数 |',
                '|---|---:|---:|---:|---:|']
            for c in ('encoder','blend05'):
                v=witnesses[c]['0.5']['early_top10']; f=v['features']['log_mask_encoder_area_ratio']
                lines.append(f"| {c} | {v['all_recovered_queries']} | {v['unique_gt_actual_recovery_witness_queries']} | {v['actual_recovery_witness_images']} | {number(f['ranking_ap'])}/{number(f['prevalence'])} |")
            lines += ['', '见证Query仍只是GT辅助的事后标签；未覆盖的归属歧义另保留，不能当成新增GT见证。剩余Query恢复可能重复覆盖已有GT、归属不明或目标转换，不能全算新增病灶。所有特征在all300/top10/top1上的见证排序统计见gt_query_correspondence.json。','']
            lines += ['### 不使用GT的Encoder有效性控制，IoU≥0.5','',
                '| 干预/子集 | Query数 | 恢复/损坏/两者覆盖 | 面积比排序AP/基数 | 与两者覆盖分布重叠 |',
                '|---|---:|---:|---:|---:|']
            for c in ('encoder','blend05'):
                for name,g in guards[c]['0.5'].items():
                    f=g['features']['log_mask_encoder_area_ratio']; m=f['metrics']['recovery_vs_all_other']
                    counts=g['counts']
                    lines.append(f"| {c}/{name} | {g['queries']} | {counts['recovered']}/{counts['damaged']}/{counts['both_good']} | {number(m['ranking_ap'])}/{number(m['prevalence'])} | {number(f['distribution_overlap']['recovery_both_good'])} |")
    lines += ['## 监督与缺额','',str(read_json(out/'training_supervision.json')),'',
        '未保存完整二值Mask或Mask logits，不能计算连通分量和空间协方差。低占用只是形状代理，不等同碎片证据。Kvasir Val没有Encoder密集选择分数/原生合法性字段，保留缺失，不拿最终分数替代。',
        '分位数分布和重叠使用GT盲的合并特征十分位分箱，不选择门控操作点。AUC保持原方向，低于0.5也照实报告；不事后翻分或挑最佳特征。缺失/单一类别时不强行估计。',
        '原生高IoU候选的保护统计、被破坏幅度>=.05、归属不明/变化、GT池恢复与原Encoder最佳Query恢复分别保留。事后GT最佳Query只用于轨迹解释。',
        '三条件全局干预共同改变self-attention，不能离线拼接分支就声称已得到自适应模型；也不计算GT Oracle AP上界。图像重采样不能证明患者级或训练seed泛化。','']
    (out/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    (out/'report_source.py').write_bytes(Path(__file__).read_bytes())
    draw(out,models)
    for path,digest in inputs.items(): assert sha256(path)==digest
    write_json(out/'report_verification.json',{'status':'passed','input_json_unchanged':True,
        'torch_imported':'torch' in __import__('sys').modules,'models_loaded':False,
        'source_sha256':sha256(Path(__file__))})


def draw(out,models):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,4,figsize=(18,13),constrained_layout=True)
    colors=['#89939c','#2166ac','#b2182b','#1b7837']
    titles=['Box–Mask disagreement','log(Mask / Encoder area)','Normalized center shift','Unfilled Mask bounding rectangle']
    for row,(model,condition) in enumerate((('baseline','encoder'),('baseline','blend05'),('mssl','encoder'),('mssl','blend05'))):
        with np.load(out/f'polypgen_{model}.query_audit.npz') as a:
            pick=a['query_id']<10
            x=a['features'][pick,:4]; labels=outcome(a['native_final_max_any_gt_iou'][pick],a[condition+'_final_max_any_gt_iou'][pick],.5)
            images=a['image_index']; w=balanced_weights(images,pick)
        for j in range(4):
            ax=axes[row,j]; values=x[:,j]
            valid=np.isfinite(values); order=np.argsort(values[valid],kind='stable')
            reference=values[valid][order]; cw=np.cumsum(w[valid][order]); cw/=cw[-1]
            # Display values on an unconditional pooled feature percentile axis.
            # This changes display scale only; no tail/gate cutoff is chosen.
            percentile=np.interp(values,reference,cw)*100
            for k,name in enumerate(CLASSES):
                m=(labels==k)&valid
                if not m.any(): continue
                idx=np.argsort(percentile[m],kind='stable'); xp=percentile[m][idx]
                yp=np.cumsum(w[m][idx]); yp/=yp[-1]
                ax.step(xp,yp,where='post',color=colors[k],label=f'{name} (n={m.sum()})',lw=1.4)
            if row==0: ax.set_title(titles[j])
            if j==0: ax.set_ylabel(f'{model}/{condition}\nWithin-class cumulative fraction')
            ax.set_xlim(-.5,100.5); ax.set_ylim(0,1.01); ax.grid(alpha=.15); ax.legend(fontsize=7,loc='lower right')
            if row==3: ax.set_xlabel('GT-blind pooled feature percentile (early Encoder top10)')
    fig.suptitle('PolypGen: recovery and protected-candidate overlap at IoU 0.5\nCPU cache audit; no trained gate or inference cutoff',fontsize=14)
    fig.savefig(out/'feature_overlap.png',dpi=150)
    fig.savefig(out/'feature_overlap.pdf')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',required=True)
    run(parser.parse_args().output)
