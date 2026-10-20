"""Freeze an outcome-blind PolypGen sample and source label counts; CPU only."""
import argparse
from collections import Counter,defaultdict
import csv
import datetime as dt
from pathlib import Path
import numpy as np
from polypbench.io import read_json,write_json,sha256


def prepare(workspace,output):
    root=Path(workspace).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    gtpath=root/'data/polypgen_zeroshot_stills_v2/provided_boxes.coco.json'
    csvpath=root/'runs/polypgen-local-relation-primary-20261021/per_gt.csv'
    trainpath=root/'data/kvasir_official_bbox_coco/train.json'
    gt=read_json(gtpath);groups=defaultdict(list)
    for ann in gt['annotations']:groups[ann['image_id']].append(ann)
    # Only GT count and already defined image/GT-derived bins are read. No
    # model score, predicted box, failure group or AP enters sample selection.
    bins={}
    with csvpath.open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['image_gt_count']=='1':
                bins[int(row['image_id'])]=(row['low_contrast']=='True',row['small']=='True')
    rng=np.random.default_rng(20261021);selected=[];shortfalls=[]
    for center in sorted({im['center'] for im in gt['images']}):
        images=sorted([im for im in gt['images'] if im['center']==center],key=lambda im:im['id'])
        pools={'negative':[im for im in images if not groups[im['id']]],
               'multi':[im for im in images if len(groups[im['id']])>1]}
        quotas={'negative':1,'multi':3}
        for low in (True,False):
            for small in (True,False):
                name=f'single_low={low}_small={small}';quotas[name]=2
                pools[name]=[im for im in images if bins.get(im['id'])==(low,small)]
        chosen=[];used=set()
        for name,count in quotas.items():
            pool=pools[name];number=min(count,len(pool))
            indices=rng.choice(len(pool),size=number,replace=False) if number else []
            for index in indices:
                im=pool[int(index)];chosen.append({**im,'sampling_stratum':name,'gt_count':len(groups[im['id']])});used.add(im['id'])
            if number<count:shortfalls.append({'center':center,'stratum':name,'requested':count,'available':len(pool)})
        remaining=[im for im in images if im['id'] not in used]
        for index in rng.choice(len(remaining),size=12-len(chosen),replace=False):
            im=remaining[int(index)];chosen.append({**im,'sampling_stratum':'same_center_uniform_fallback','gt_count':len(groups[im['id']])})
        assert len(chosen)==12 and len({im['id'] for im in chosen})==12
        selected.extend(chosen)
    train=read_json(trainpath);train_images={im['id']:im for im in train['images']};tg=defaultdict(list)
    for ann in train['annotations']:tg[ann['image_id']].append(ann)
    ratios=[a['bbox'][2]*a['bbox'][3]/(train_images[a['image_id']]['height']*train_images[a['image_id']]['width']) for a in train['annotations']]
    area_cutoff=read_json(root/'runs/polypgen-local-relation-primary-20261021/audit.json')['small_cutoff']
    counts={'images':len(train_images),'gt':len(ratios),'images_by_gt_count':dict(sorted(Counter(len(tg[i]) for i in train_images).items())),
        'single_gt':sum(len(tg[i])==1 for i in train_images),'multi_gt':sum(len(tg[i]) for i in train_images if len(tg[i])>1),
        'relative_area_q25':float(np.quantile(ratios,.25)),
        'gt_under_polypgen_area_q25':sum(r<=area_cutoff for r in ratios),'polypgen_area_q25_reference':area_cutoff,
        'limits':'Original boxes only; counts do not establish instance masks or survival under LSJ. Area threshold is a descriptive target benchmark reference, not a training hyperparameter.'}
    protocol={'created_at':dt.datetime.now(dt.timezone.utc).isoformat(),'status':'prepared_not_gpu_executed',
        'sample_size':72,'images_per_center':12,'seed':20261021,'selection_uses_model_outputs':False,
        'selection':'Per center: 1 negative, 3 multi, 2 per single low/nonlow contrast x small/nonsmall cell; shortage filled uniformly from remaining same-center images',
        'sampling_shortfalls':shortfalls,'sample_population_claim':False,
        'cohort_role':'PolypGen is the primary performance benchmark and already observed for method development; not blind-test evidence',
        'models':['frozen_baseline','frozen_mssl_v1'],'training_updates':False,'runtime_location':'A6000 training machine; do not add GPU jobs to local E/F machine',
        'stages':['legal_dense_encoder_boxes','native_top300_encoder_boxes','actual_mask2box_decoder_init',
                  'ordinary_decoder_aux_predictions_in_order','final_decoder300','native_final_top100'],
        'thresholds':[.5,.75,.9],'coverage':'one-to-one maximum cardinality at each stage, shared original-image coordinates',
        'tracking':'same query identity within each model through init/decoder; align models by GT, not equal numeric query IDs',
        'extra_diagnostics':['invalid/padding proposal exclusion','empty/degenerate predicted-mask initialization',
            'GT support tokens and feature margins where legal','stage gain/loss transitions; no unique causal loss decomposition'],
        'parity_gate':'Final boxes, scores and query IDs must agree with frozen native image-only inference; no GT to model; eval/no_grad; no matcher/TopK replacements',
        'source_label_audit_required':['original small/multi distribution','raw semantic-mask provenance and unique-box spatial support',
            'fixed LSJ replay, valid/padding/retained GT regions','coverage per prospective feature level, legitimate shortages'],
        'source_label_counts':counts,
        'inputs_sha256':{str(p):sha256(p) for p in (Path(__file__),gtpath,csvpath,trainpath)},
        'decision':'First localize stage-specific failure and training-supervision limits. Do not freeze an encoder contrast objective or start full new training from final300 coverage alone.'}
    write_json(out/'sample.images.json',{'images':selected})
    write_json(out/'protocol.json',protocol);write_json(out/'source_label_counts.json',counts)
    print({'images':len(selected),'centers':6,'shortfalls':shortfalls,'source_label_counts':counts})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args();prepare(a.workspace,a.output)
