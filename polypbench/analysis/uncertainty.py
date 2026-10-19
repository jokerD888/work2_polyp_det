"""Paired, centre-stratified image bootstrap with exact COCO accumulation."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from polypbench.io import write_json
from .coco_cache import CocoCache,duplicate_dataset,official_eval,ap_vector,metrics

_CACHES=None
_REGIONS=None


def _bootstrap_one(index,counts):
    result={}
    for label,cache in _CACHES.items():
        result[label]={"pooled":cache.accumulate(counts)}
        for center,(mask,regional) in _REGIONS[label].items():
            result[label][center]=regional.accumulate(counts[mask])
    return index,result


def interval(values):
    low,high=np.quantile(values,[.025,.975])
    return {"low_pp":float(low),"high_pp":float(high),"contains_zero":bool(low<=0<=high)}


def run(gt,predictions,output,replicates=2000,seed=20261019,workers=4,draws_path=None):
    global _CACHES,_REGIONS
    out=Path(output); out.mkdir(parents=True,exist_ok=False)
    caches={label:CocoCache(gt,rows) for label,rows in predictions.items()}
    ids=next(iter(caches.values())).ids; image={im["id"]:im for im in gt["images"]}
    centers=sorted({im["center"] for im in gt["images"]}); masks={c:np.array([image[int(i)]["center"]==c for i in ids]) for c in centers}
    if draws_path is None:
        rng=np.random.default_rng(seed); draws=np.zeros((replicates,len(ids)),dtype=np.uint16)
        for c,mask in masks.items():
            n=int(mask.sum()); draws[:,mask]=rng.multinomial(n,np.full(n,1/n),size=replicates)
    else:
        draws=np.load(draws_path,allow_pickle=False)
        if draws.shape!=(replicates,len(ids)) or draws.dtype.kind not in 'iu' or (draws<0).any():
            raise ValueError("Reused image multiplicities have invalid dimensions/type")
        for c,mask in masks.items():
            if not np.all(draws[:,mask].sum(1)==int(mask.sum())):
                raise ValueError(f"Reused draws changed center composition: {c}")
    np.save(out/"image_multiplicities.npy",draws)
    verification={"same_multiset_for_all_models":True,"fresh_copy_image_ids":"canonical original-ID order, then copy index; new annotation IDs too",
                  "accumulation":"cached per-image official COCO matches, re-expanded score order and 101-point precision/recall each replicate; not mean per-image AP",
                  "scope":"one category, bbox, all-area, maxDets100, native scores; COCO dtIgnore/gtIgnore retained",
                  "official_parity":{}}
    point={}; loo={c:{} for c in centers}; sensitivity={}
    for label,cache in caches.items():
        np.testing.assert_allclose(cache.accumulate(np.ones(len(ids),int)),cache.reference,rtol=0,atol=1e-12)
        errors=[]
        # Verify real full-cohort resamples including duplicate image copies,
        # not just synthetic fixtures. No outcomes determine the sample draws.
        for rep in (0,1):
            sampled=np.repeat(ids,draws[rep]).tolist()
            cloned,rows=duplicate_dataset(gt,predictions[label],sampled)
            expected=ap_vector(official_eval(cloned,rows)); actual=cache.accumulate(draws[rep])
            np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
            errors.append(float(np.max(abs(actual-expected))))
        verification["official_parity"][label]={"full_cohort_and_first_two_resamples":"passed","max_absolute_ap_error":max(errors)}
        point[label]={"pooled":metrics(cache.reference)}
        for c,mask in masks.items():
            point[label][c]=metrics(cache.accumulate(mask.astype(int)))
            loo[c][label]=metrics(cache.accumulate((~mask).astype(int)))
    write_json(out/"cache_verification.json",verification)
    _CACHES=caches
    _REGIONS={label:{c:(mask,cache.restrict(mask)) for c,mask in masks.items()} for label,cache in caches.items()}
    samples={label:{c:np.empty((replicates,10)) for c in ["pooled",*centers]} for label in predictions}
    start=time.monotonic(); done=0
    # Only CPU workers; no agent delegation and no model training.
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context("fork")) as pool:
        futures=[pool.submit(_bootstrap_one,i,draws[i]) for i in range(replicates)]
        for future in as_completed(futures):
            i,result=future.result()
            for label,cohorts in result.items():
                for c,vector in cohorts.items(): samples[label][c][i]=vector
            done+=1
            if done%50==0 or done==replicates:
                elapsed=time.monotonic()-start
                write_json(out/"progress.json",{"replicates_done":done,"replicates_total":replicates,
                    "elapsed_seconds":elapsed,"eta_seconds":elapsed/done*(replicates-done)})
    np.savez_compressed(out/"bootstrap_ap_vectors.npz",**{f"{label}__{c}":a for label,cc in samples.items() for c,a in cc.items()})
    pairs={}; model_intervals={}
    def extract(a):
        return {"AP":a.mean(1),"AP75":a[:,5],"AP80":a[:,6],"AP90":a[:,8],"AP95":a[:,9],"AP80_95_mean":a[:,6:].mean(1)}
    for label,cc in samples.items():
        model_intervals[label]={c:{k:interval(v*100) for k,v in extract(a).items()} for c,a in cc.items()}
    for control in (label for label in predictions if label != "mssl"):
        pairs[control]={}
        for c in ["pooled",*centers]:
            differences=extract(samples["mssl"][c]-samples[control][c])
            pairs[control][c]={k:{"point_delta_pp":100*(point["mssl"][c][k]-point[control][c][k]),
                                  **interval(v*100)} for k,v in differences.items()}
    result={"protocol":{"replicates":replicates,"bootstrap_seed":seed,"unit":"image; sampled within each observed center, paired across all supplied models",
        "reused_draws":str(draws_path) if draws_path else None,
        "center_composition":"fixed at original counts in each replicate; conditional on the six observed centers",
        "limitations":"No verified patient/video mapping. Image dependence may make intervals optimistic; not patient-level uncertainty or training-seed uncertainty. Multiple metric intervals are exploratory, not multiplicity-adjusted hypothesis tests.",
        "interval":"percentile 2.5/97.5; AP recalculated from every resampled multiset"},
        "point_fraction":point,"paired_intervals":pairs,"model_intervals":model_intervals,
        "leave_one_center_out_fraction":loo}
    write_json(out/"uncertainty.json",result)
    return result
