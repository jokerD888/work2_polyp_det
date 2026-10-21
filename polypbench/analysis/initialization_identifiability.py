"""CPU-only audit of frozen caches; no inference, fitting or gate selection.

Features accept ONLY pre-decoder fields. GT labels are constructed separately.
All 300 queries are retained. Correlated queries receive equal total image
weight, with original counts and image denominators reported alongside it.
"""
import argparse
from collections import Counter, defaultdict
import datetime as dt
from pathlib import Path
import time

import numpy as np

from polypbench.io import read_json, sha256, write_json

THRESHOLDS = (.5, .75, .9)
CLASSES = ("both_bad", "recovered", "damaged", "both_good")
FEATURES = ("box_mask_disagreement", "log_mask_encoder_area_ratio", "normalized_center_shift",
    "mask_bbox_unfilled_fraction", "mask_foreground_fraction", "mask_padding_foreground_fraction",
    "mask_empty", "mask_degenerate", "encoder_boundary", "encoder_area", "mask_area",
    "encoder_selection_rank", "encoder_low_selection_score", "encoder_not_native_legal")
PRE_FIELDS = ("init_encoder_normalized", "init_mask_normalized", "mask2box_raw_lowres_xyxy",
    "mask_foreground_pixels", "mask_foreground_fraction", "mask_padding_foreground_fraction",
    "mask_empty", "mask_one_row", "mask_one_column")


def before_decoder_features(pre, selection_scores=None, legal=None):
    """No GT, final boxes, final scores or GT-selected Query is accepted here."""
    e = np.asarray(pre["init_encoder_normalized"], dtype=np.float64)
    m = np.asarray(pre["init_mask_normalized"], dtype=np.float64)
    assert e.shape == m.shape == (300,4) and np.isfinite(e).all() and np.isfinite(m).all()
    ea, ma = e[:,2:].prod(1), m[:,2:].prod(1)
    elo, ehi = e[:,:2]-e[:,2:]/2, e[:,:2]+e[:,2:]/2
    mlo, mhi = m[:,:2]-m[:,2:]/2, m[:,:2]+m[:,2:]/2
    inter = np.maximum(np.minimum(ehi,mhi)-np.maximum(elo,mlo),0).prod(1)
    overlap = inter/np.maximum(ea+ma-inter,1e-12)
    raw = np.asarray(pre["mask2box_raw_lowres_xyxy"], dtype=np.float64)
    # masks_to_boxes stores inclusive integer extrema, unlike continuous box area.
    rectangle = np.maximum(raw[:,2:]-raw[:,:2]+1,0).prod(1)
    empty = np.asarray(pre["mask_empty"], dtype=bool)
    occupancy = np.divide(pre["mask_foreground_pixels"], rectangle,
        out=np.full(300,np.nan), where=(rectangle>0)&~empty)
    values = [1-overlap, np.log(np.maximum(ma,1e-12)/np.maximum(ea,1e-12)),
        np.linalg.norm((m[:,:2]-e[:,:2])/np.maximum(e[:,2:],1e-9),axis=1),
        1-occupancy, pre["mask_foreground_fraction"], pre["mask_padding_foreground_fraction"],
        empty, np.asarray(pre["mask_one_row"])|np.asarray(pre["mask_one_column"]),
        ((e<=0)|(e>=1)).any(1), ea, ma, np.arange(1,301),
        1-np.asarray(selection_scores) if selection_scores is not None else np.full(300,np.nan),
        ~np.asarray(legal,dtype=bool) if legal is not None else np.full(300,np.nan)]
    return np.stack(values,axis=1).astype(np.float64)


def outcome(native, alternative, threshold):
    n, a = np.asarray(native)>=threshold, np.asarray(alternative)>=threshold
    # 0 both_bad, 1 recovered, 2 damaged, 3 both_good
    return np.where(n, np.where(a,3,2), np.where(a,1,0)).astype(np.uint8)


def ownership(matrices):
    """GT-assisted bookkeeping ONLY; never part of feature extraction/selection."""
    n = matrices[0].shape[0]
    if not n:
        return np.full(300,-1,dtype=np.int16), np.zeros(300,bool), np.zeros(300,bool)
    compatible = np.maximum.reduce(matrices)>=.5
    count = compatible.sum(0)
    owner = np.where(count==1,compatible.argmax(0),np.where(count==0,-1,-2)).astype(np.int16)
    def switches(items):
        best = np.stack([np.where(m.max(0)>=.5,m.argmax(0),-1) for m in items])
        lo = np.where(best>=0,best,n+1).min(0)
        return (best.max(0)>=0)&(lo<best.max(0))
    initial_switch = switches(matrices[:2])
    final_switch = switches(matrices[2:])
    return owner, initial_switch, final_switch


def balanced_weights(image_index, mask):
    counts = np.bincount(image_index[mask],minlength=int(image_index.max())+1)
    return 1/np.maximum(counts[image_index[mask]],1)


def weighted_quantiles(x, w, probabilities):
    valid = np.isfinite(x)&np.isfinite(w)&(w>0)
    if not valid.any():
        return [None]*len(probabilities)
    x,w=x[valid],w[valid]; order=np.argsort(x,kind="stable"); x,w=x[order],w[order]
    return np.interp(probabilities,(np.cumsum(w)-.5*w)/w.sum(),x).tolist()


def rank_metrics(x, y, weights):
    """Fixed HIGHER score predicts recovery; never flip to maximize observed AUC.

    Ties are grouped. AP is ranking average precision, NOT detection bbox AP.
    """
    valid=np.isfinite(x)&(weights>0)
    x,y,w=np.asarray(x)[valid],np.asarray(y,dtype=bool)[valid],np.asarray(weights)[valid]
    positive, negative=w[y].sum(),w[~y].sum()
    if positive<=0 or negative<=0:
        return {"auc":None,"ranking_ap":None,"prevalence":float(positive/w.sum()) if w.sum() else None}
    order=np.argsort(x,kind="stable"); x,y,w=x[order],y[order],w[order]
    starts=np.r_[0,np.flatnonzero(x[1:]!=x[:-1])+1]
    p=np.add.reduceat(w*y,starts); n=np.add.reduceat(w*~y,starts)
    auc=np.sum(p*(np.cumsum(n)-n+.5*n))/(positive*negative)
    p,n=p[::-1],n[::-1]
    precision=np.cumsum(p)/np.cumsum(p+n)
    ap=np.sum((p/positive)*precision)
    return {"auc":float(auc),"ranking_ap":float(ap),"prevalence":float(positive/(positive+negative))}


def feature_statistics(x, labels, images, weights, normal, uncertainty=False, centers=None):
    result={}
    for j,name in enumerate(FEATURES):
        values=x[:,j]; valid=np.isfinite(values)
        distributions={}
        edges=np.unique(weighted_quantiles(values,weights,np.linspace(0,1,11))) if valid.any() else np.array([])
        histograms={}
        for c,title in enumerate(CLASSES):
            pick=(labels==c)&valid
            hist=np.histogram(values[pick],bins=edges,weights=weights[pick])[0] if len(edges)>1 else np.array([weights[pick].sum()])
            histograms[title]=hist/hist.sum() if hist.sum() else np.zeros_like(hist)
            distributions[title]={"queries":int((labels==c).sum()),"finite_queries":int(pick.sum()),
                "images":int(len(np.unique(images[pick]))),"image_balanced_mass":float(weights[pick].sum()),
                "quantiles_10_25_50_75_90":weighted_quantiles(values[pick],weights[pick],[.1,.25,.5,.75,.9])}
        def overlap(a,b):
            if not histograms[a].sum() or not histograms[b].sum(): return None
            return float(np.minimum(histograms[a],histograms[b]).sum())
        pair=(labels==1)|(labels==2)
        protected=(labels==1)|normal
        metrics={"recovery_vs_all_other":rank_metrics(values,labels==1,weights),
            "recovery_vs_damage":rank_metrics(values[pair],labels[pair]==1,weights[pair]),
            "recovery_vs_native_covered":rank_metrics(values[protected],labels[protected]==1,weights[protected])}
        item={"missing_queries":int((~valid).sum()),"distributions":distributions,
            "higher_means_predicted_recovery":True,"metrics":metrics,
            "pooled_gt_blind_histogram_edges":edges.tolist(),
            "distribution_overlap":{"recovery_damage":overlap("recovered","damaged"),
                "recovery_both_good":overlap("recovered","both_good"),
                "recovery_both_bad":overlap("recovered","both_bad")},
            "native_covered_control":{"queries":int(normal.sum()),
                "images":int(len(np.unique(images[normal]))),
                "quantiles_10_50_90":weighted_quantiles(values[normal],weights[normal],[.1,.5,.9])}}
        # Fixed three cues, 100 image-block resamples; no feature selection or new cutoffs.
        if uncertainty and name in FEATURES[:3] and valid.any():
            rng=np.random.default_rng(42); unique=np.unique(images)
            values_boot=[]
            blocks=[unique] if centers is None else [np.unique(images[centers==c]) for c in np.unique(centers)]
            for _ in range(100):
                sampled=np.concatenate([rng.choice(block,len(block),replace=True) for block in blocks if len(block)])
                counts=np.bincount(sampled,minlength=int(images.max())+1)
                metric=rank_metrics(values,labels==1,weights*counts[images])
                if metric["ranking_ap"] is not None: values_boot.append(metric["ranking_ap"])
            item["image_block_bootstrap_ranking_ap_95_interval"]=(np.quantile(values_boot,[.025,.975]).tolist() if values_boot else None)
        result[name]=item
    return result


def subset_report(features, labels, images, selected, detailed=True, bootstrap=False, centers=None):
    ids=np.flatnonzero(selected)
    if not len(ids): return {"queries":0,"images":0}
    y=labels[ids]; image_ids=images[ids]; w=balanced_weights(images,selected)
    normal=(y==2)|(y==3)
    result={"queries":len(ids),"images":int(len(np.unique(image_ids))),
        "counts":dict(zip(CLASSES,np.bincount(y,minlength=4).astype(int).tolist())),
        "image_balanced_class_fraction":{name:float(w[y==c].sum()/w.sum()) for c,name in enumerate(CLASSES)}}
    if detailed:
        result["features"]=feature_statistics(features[ids],y,image_ids,w,normal,bootstrap,
            centers[ids] if centers is not None else None)
    return result


def audit_one(root, cohort, model, out, record_input, progress):
    directory=root/f"{cohort}_{model}"
    verification=read_json(record_input(directory/"verification.json"))
    assert verification["status"]=="passed" and verification["upstream_bitwise_identity_all_conditions"]
    job=read_json(record_input(root/f"{cohort}_{model}.job.json"))
    samples=read_json(record_input(Path(job["sample"])))['images']
    native_reports=[]; blocks=[]; owners=[]; initial_switches=[]; final_switches=[]
    native_ious=[]; alternate_ious={c:[] for c in ("encoder","blend05")}
    image_indices=[]; image_centers=[]; multiplicities=[]; gt_sizes=[]; legal_flags=[]; gt_rows=[]
    for index,image in enumerate(samples):
        iid=image['id']
        report=read_json(record_input(directory/"native/images"/f"{iid}.json")); native_reports.append(report)
        files={c:directory/c/"images"/f"{iid}.npz" for c in ("native","encoder","blend05")}
        data={}
        for c,path in files.items():
            with np.load(record_input(path)) as archive:
                keys=(PRE_FIELDS+("top300_token_ids","encoder_top300_gt_iou","decoder_init_gt_iou","decoder_09_gt_iou","final_top100_gt_iou")) if c=="native" else ("top300_token_ids","decoder_09_gt_iou","final_top100_gt_iou")
                data[c]={k:archive[k] for k in keys}
        n=data['native']; scores=legal=None
        if cohort=='polypgen':
            historical=Path(job['historical_stages'])/"images"/f"{iid}.npz"
            assert sha256(historical)==job['historical_stage_sha256'][str(iid)]
            with np.load(record_input(historical)) as a:
                np.testing.assert_array_equal(a['top300_token_ids'],n['top300_token_ids'])
                scores=a['dense_class_scores'][n['top300_token_ids']]
                legal=a['dense_native_legal'][n['top300_token_ids']]
        blocks.append(before_decoder_features({k:n[k] for k in PRE_FIELDS},scores,legal))
        mats=[n['encoder_top300_gt_iou'],n['decoder_init_gt_iou']]+[data[c]['decoder_09_gt_iou'] for c in ("native","encoder","blend05")]
        owner,init_switch,final_switch=ownership(mats)
        owners.append(owner); initial_switches.append(init_switch); final_switches.append(final_switch)
        sizes=np.zeros(300,dtype=np.int8)
        for q in np.flatnonzero(owner>=0):
            sizes[q]=int(report['gt_metadata'][owner[q]]['relative_area_quartile'][1:])
        gt_sizes.append(sizes)
        image_indices.append(np.full(300,index,dtype=np.int32)); image_centers.append(np.full(300,image.get('center','Kvasir')))
        multiplicities.append(np.full(300,min(report['gt'],2),dtype=np.int8))
        legal_flags.append(legal if legal is not None else np.full(300,False))
        for c in ("encoder","blend05"):
            np.testing.assert_array_equal(n['top300_token_ids'],data[c]['top300_token_ids'])
        for c,matrix in zip(("native","encoder","blend05"),mats[2:]):
            values=matrix.max(0) if len(matrix) else np.zeros(300)
            if c=='native': native_ious.append(values)
            else: alternate_ious[c].append(values)
        for j,aid in enumerate(report['annotation_ids']):
            q0=int(n['encoder_top300_gt_iou'][j].argmax())  # GT privilege, never features or query selection
            for c in ("encoder","blend05"):
                row={"image_id":iid,"annotation_id":aid,"condition":c,"center":image.get('center','Kvasir'),
                    "gt_selected_encoder_query_diagnostic_only":q0,"multiplicity":'single' if report['gt']==1 else 'multi',
                    **report['gt_metadata'][j],"stages":{}}
                for stage in ('decoder_09','final_top100'):
                    a,b=n[stage+'_gt_iou'][j],data[c][stage+'_gt_iou'][j]
                    row['stages'][stage]={"native":float(a.max()),"alternative":float(b.max()),
                        "events":{str(t):CLASSES[int(outcome(a.max(),b.max(),t))] for t in THRESHOLDS}}
                a,b=n['decoder_09_gt_iou'][j],data[c]['decoder_09_gt_iou'][j]
                row['same_original_encoder_query_final_iou']={"native":float(a[q0]),"alternative":float(b[q0])}
                row['native_final_best_query_diagnostic_only']=int(a.argmax())
                row['alternative_final_best_query_diagnostic_only']=int(b.argmax())
                row['continuous_final_max_iou_delta']=float(b.max()-a.max())
                gt_rows.append(row)
        progress(index+1,len(samples))
    assert len(samples)==verification['images'] and sum(r['gt'] for r in native_reports)==verification['gt']
    x=np.concatenate(blocks); owner=np.concatenate(owners); image_ids=np.concatenate(image_indices)
    centers=np.concatenate(image_centers); multi=np.concatenate(multiplicities); sizes=np.concatenate(gt_sizes)
    native=np.concatenate(native_ious); initial_switch=np.concatenate(initial_switches); final_switch=np.concatenate(final_switches)
    legal=np.concatenate(legal_flags); ranks=np.tile(np.arange(300),len(samples))
    base_subsets={"all300":np.ones(len(x),bool),"early_top10":ranks<10,
        "early_top1":ranks==0,"encoder_interior":x[:,FEATURES.index('encoder_boundary')]==0}
    if cohort=='polypgen': base_subsets['native_legal_encoder']=legal
    result={"images":len(samples),"gt":verification['gt'],"queries":len(x),
        "ownership":{"unique_gt":int((owner>=0).sum()),"unassigned":int((owner==-1).sum()),
            "ambiguous":int((owner==-2).sum()),"initial_gt_disagreement":int(initial_switch.sum()),
            "final_gt_switch":int(final_switch.sum()),"switch_counts_overlap_ambiguous":True},"conditions":{}}
    npz={"features":x,"feature_names":np.array(FEATURES),"image_index":image_ids,"image_ids":np.array([i['id'] for i in samples]),
        "query_id":ranks,"centers":centers,"multiplicity":multi,"owner_gt_index":owner,
        "owner_gt_area_quartile":sizes,"initial_gt_disagreement":initial_switch,"final_gt_switch":final_switch,
        "native_final_max_any_gt_iou":native}
    for c,values in alternate_ious.items():
        alt=np.concatenate(values); npz[c+'_final_max_any_gt_iou']=alt
        condition={"thresholds":{},"strata":{}}
        for t in THRESHOLDS:
            labels=outcome(native,alt,t); subsets={}
            for name,selection in base_subsets.items():
                # Raw complete-query census is retained, not only posthoc supported objects.
                subsets[name]=subset_report(x,labels,image_ids,selection,detailed=name in ('all300','early_top10','early_top1'),
                    bootstrap=name=='early_top10' and t==.5,centers=centers)
            for name,selection in {"unique_gt_early_top10":(owner>=0)&(ranks<10),
                "ambiguous_or_switched":(owner==-2)|initial_switch|final_switch,
                "unassigned":owner==-1}.items():
                subsets[name]=subset_report(x,labels,image_ids,selection,detailed=name=='unique_gt_early_top10')
            if cohort=='polypgen':
                selection=(owner>=0)&legal
                subsets['unique_gt_native_legal']=subset_report(x,labels,image_ids,selection)
            condition['thresholds'][str(t)]=subsets
            if t in (.5,.9):
                for group,selection in {
                    **{'center='+z:(centers==z)&(ranks<10) for z in np.unique(centers)},
                    **{'multiplicity='+str(z):(multi==z)&(ranks<10) for z in (0,1,2)},
                    **{'unique_gt_area=q'+str(z):(sizes==z)&(owner>=0)&(ranks<10) for z in (1,2,3,4)},
                    **{'unique_gt_area=q'+str(z)+'/multiplicity='+str(m):(sizes==z)&(owner>=0)&(multi==m)&(ranks<10)
                       for z in (1,2,3,4) for m in (1,2)}}.items():
                    # Strata use fixed pre-top10; size strata are explicitly GT-conditioned.
                    condition['strata'].setdefault(group,{})[str(t)]=subset_report(x,labels,image_ids,selection)
        # Strict localization protection and continuous differences, no new gate thresholds.
        protected=native>=.9
        condition['native_high_precision_queries']={"queries":int(protected.sum()),
            "images":int(len(np.unique(image_ids[protected]))),"alternative_still_ge_09":int((protected&(alt>=.9)).sum()),
            "alternative_below_075":int((protected&(alt<.75)).sum()),
            "iou_drop_ge_005":int((protected&(alt<=native-.05)).sum())}
        result['conditions'][c]=condition
    gt_summary={}
    for c in ('encoder','blend05'):
        items=[r for r in gt_rows if r['condition']==c]
        summary={}
        for stage in ('decoder_09','final_top100'):
            summary[stage]={}
            for t in THRESHOLDS:
                counts=Counter(r['stages'][stage]['events'][str(t)] for r in items)
                summary[stage][str(t)]={k:counts[k] for k in CLASSES}
        recovered=[r for r in items if r['stages']['decoder_09']['events']['0.5']=='recovered']
        summary['recovered_final300_gt']={"gt":len(recovered),
            "original_best_encoder_query_now_covered":sum(r['same_original_encoder_query_final_iou']['alternative']>=.5 for r in recovered),
            "best_query_id_changed":sum(r['native_final_best_query_diagnostic_only']!=r['alternative_final_best_query_diagnostic_only'] for r in recovered),
            "GT_selected_query_is_not_deployable_rule":True}
        gt_summary[c]=summary
    result['gt_census']=gt_summary
    np.savez_compressed(out/f"{cohort}_{model}.query_audit.npz",**npz)
    write_json(out/f"{cohort}_{model}.per_gt.json",gt_rows)
    write_json(out/f"{cohort}_{model}.json",result)
    # Exact consistency with the already completed full intervention, not just a new analysis.
    previous=read_json(record_input(directory/'comparison.json'))
    for c in ('encoder','blend05'):
        for stage in ('decoder_09','final_top100'):
            for t in THRESHOLDS:
                counts=gt_summary[c][stage][str(t)]; expected=previous[c]['all']['stages'][stage][str(t)]
                assert counts['recovered']==expected['recovered'] and counts['damaged']==expected['newly_lost']
    return result


def run(workspace, traces, output):
    workspace,root,out=map(lambda p:Path(p).resolve(),(workspace,traces,output))
    out.mkdir(parents=True,exist_ok=False)
    assert read_json(root/'state.json')['status']=='completed'
    protocol={"created_at":dt.datetime.now(dt.timezone.utc).isoformat(),"cpu_only":True,"model_inference":False,
        "fitting":False,"gate_or_coefficient_search":False,"features":FEATURES,"feature_orientation":"higher predicts recovery; never flip based on AUC",
        "cohorts":["polypgen1473/1533","kvasir_val120/130"],"conditions":["encoder","blend05"],"thresholds":THRESHOLDS,
        "subsets":"all300; native Encoder top10/top1 (TopK order before decoder, no GT); nonboundary encoder; native-legal encoder where available",
        "GT_conditioning":"unique/ambiguous/unassigned ownership and GT size are diagnosis-only; all-query census remains primary",
        "uncertainty":"100 seed42 image-block resamples stratified by center, ranking AP for first three fixed features on early-top10 .5; no patient/seed inference",
        "normal_controls":"both_good + damaged are native-covered controls; large both_bad/unassigned background base retained",
        "missing":"No full mask/connected components/spatial covariance; Kvasir selection score and native-legal flags absent, NaN not fabricated",
        "limits":"Full-query interventions alter self-attention jointly; do not combine cached query branches into a purported real gated model. Ranking AP is not bbox AP. Associations do not validate a deployable gate.",
        "source_sha256":{rel:sha256(workspace/rel) for rel in ('polypbench/analysis/initialization_identifiability.py',)}}
    write_json(out/'protocol.json',protocol)
    (out/'source.py').write_bytes((workspace/'polypbench/analysis/initialization_identifiability.py').read_bytes())
    hashes={}
    def record_input(path):
        path=Path(path); key=str(path)
        digest=sha256(path)
        if key in hashes: assert hashes[key]==digest
        else: hashes[key]=digest
        return path
    start=time.monotonic(); state={"status":"running","stage":"audit","completed":[]}
    def save():
        state['elapsed_seconds']=time.monotonic()-start; write_json(out/'state.json',state)
    summaries={}
    try:
        for cohort in ('polypgen','kvasir_val'):
            for model in ('baseline','mssl'):
                name=cohort+'_'+model; state['stage']=name; save()
                def progress(done,total):
                    state.update(images_done=done,images_total=total); save()
                summaries[name]=audit_one(root,cohort,model,out,record_input,progress)
                state['completed'].append(name); save()
        train=read_json(record_input(workspace/'data/kvasir_official_bbox_coco/train.json'))
        annotations=defaultdict(list)
        for a in train['annotations']: annotations[a['image_id']].append(a)
        supervision={"images":len(train['images']),"boxes":len(train['annotations']),
            "single_images":sum(len(annotations[i['id']])==1 for i in train['images']),
            "multi_images":sum(len(annotations[i['id']])>1 for i in train['images']),
            "limits":"Provided boxes can define quality comparisons only after a defensible GT/Query association. Current train-query prediction features and alternative outcomes were NOT cached here. Actual learnability on Train is unverified; binary semantic masks do not supply arbitrary instance identities."}
        write_json(out/'training_supervision.json',supervision)
        state['stage']='input_recheck'; save()
        for path,digest in hashes.items():
            if sha256(path)!=digest: raise RuntimeError('Input changed: '+path)
        for rel,digest in protocol['source_sha256'].items():
            assert sha256(workspace/rel)==digest
        write_json(out/'verification.json',{"status":"passed","inputs_unchanged":True,"input_files":len(hashes),
            "input_sha256":hashes,"cpu_only":True,"torch_imported":'torch' in __import__('sys').modules,
            "models_or_optimizers_loaded":False,"GT_counts_match_prior_intervention":True,
            "elapsed_seconds":time.monotonic()-start})
        state.update(status='completed',stage='audit_complete'); save()
    except BaseException as error:
        state.update(status='failed',error=repr(error)); save(); raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('workspace','traces','output'): parser.add_argument('--'+key,required=True)
    run(**vars(parser.parse_args()))
