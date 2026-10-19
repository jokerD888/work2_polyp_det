"""Exact per-image COCO matches, re-accumulated for independent image copies.

Matching is independent across images. Resampling may therefore reuse those
matches, but must re-sort scores and recompute precision/recall, not average AP.
Copies are assigned IDs in canonical original-image order. Equal-score runs
within the same image must be tiled per copy, not repeated detection-by-detection.
"""
import contextlib
import copy
import io

import numpy as np


def duplicate_dataset(gt, predictions, sampled_ids):
    """Materialize a bootstrap multiset with fresh image/annotation IDs."""
    images={x["id"]:x for x in gt["images"]}; anns={i:[] for i in images}; pred={i:[] for i in images}
    for a in gt["annotations"]: anns[a["image_id"]].append(a)
    for p in predictions: pred[p["image_id"]].append(p)
    result={k:copy.deepcopy(v) for k,v in gt.items() if k not in ("images","annotations")}
    result.update(images=[],annotations=[]); rows=[]
    for new_id,old_id in enumerate(sorted(sampled_ids),1):
        image=copy.deepcopy(images[old_id]); image["id"]=new_id; result["images"].append(image)
        for a in anns[old_id]:
            a=copy.deepcopy(a); a.update(id=len(result["annotations"])+1,image_id=new_id); result["annotations"].append(a)
        for p in pred[old_id]:
            p=copy.deepcopy(p); p["image_id"]=new_id; rows.append(p)
    return result,rows


def official_eval(gt, predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        coco=COCO(); coco.dataset=copy.deepcopy(gt); coco.createIndex()
        if predictions:
            dt=coco.loadRes(copy.deepcopy(predictions))
        else:
            dt=COCO(); dt.dataset={"images":copy.deepcopy(gt["images"]),"categories":copy.deepcopy(gt["categories"]),"annotations":[]}; dt.createIndex()
        ev=COCOeval(coco,dt,"bbox"); ev.params.imgIds=sorted(coco.imgs); ev.params.catIds=sorted(coco.cats)
        ev.params.maxDets=[1,10,100]; ev.evaluate(); ev.accumulate()
    return ev


def ap_vector(ev):
    p=ev.eval["precision"][:,:,:,0,2]
    return np.array([x[x>=0].mean() if np.any(x>=0) else np.nan for x in p])


class CocoCache:
    def __init__(self,gt,predictions):
        if len(gt["categories"])!=1: raise ValueError("Analysis is validated for one category")
        self.ids=np.array(sorted(x["id"] for x in gt["images"]),dtype=np.int64)
        ev=official_eval(gt,predictions); entries={}
        for entry in ev.evalImgs:
            if entry is not None and entry["aRng"]==ev.params.areaRng[0]: entries[entry["image_id"]]=entry
        scores=[]; image_index=[]; matches=[]; ignored=[]; counts=[]
        for i,iid in enumerate(self.ids):
            e=entries.get(int(iid))
            counts.append(int((~np.asarray(e["gtIgnore"],dtype=bool)).sum()) if e is not None else 0)
            if e is None: continue
            n=len(e["dtScores"])
            scores.extend(e["dtScores"]); image_index.extend([i]*n)
            matches.append(e["dtMatches"]>0); ignored.append(e["dtIgnore"])
        order=np.argsort(-np.asarray(scores),kind="mergesort")
        self.scores=np.asarray(scores)[order]; self.image_index=np.asarray(image_index,dtype=np.int32)[order]
        self.matches=np.concatenate(matches,axis=1)[:,order] if matches else np.zeros((10,0),bool)
        self.ignored=np.concatenate(ignored,axis=1)[:,order] if ignored else np.zeros((10,0),bool)
        self.gt_counts=np.array(counts,dtype=np.int32)
        self.reference=ap_vector(ev)
        # Only multi-detection equal-score runs within one original image need
        # corrected expansion: copy1 [a,b], copy2 [a,b], rather than [a,a,b,b].
        boundaries=np.r_[0,np.flatnonzero((self.scores[1:]!=self.scores[:-1])|
                            (self.image_index[1:]!=self.image_index[:-1]))+1,len(order)]
        self.tie_runs=[(int(a),int(b)) for a,b in zip(boundaries[:-1],boundaries[1:]) if b-a>1]

    def restrict(self,selection):
        """Keep the same per-image matches for a fixed subset of image IDs."""
        selection=np.asarray(selection,dtype=bool)
        obj=object.__new__(CocoCache); obj.ids=self.ids[selection]; obj.gt_counts=self.gt_counts[selection]
        lookup=np.full(len(self.ids),-1,dtype=np.int32); lookup[selection]=np.arange(selection.sum())
        keep=selection[self.image_index]
        obj.image_index=lookup[self.image_index[keep]]; obj.scores=self.scores[keep]
        obj.matches=self.matches[:,keep]; obj.ignored=self.ignored[:,keep]
        boundaries=np.r_[0,np.flatnonzero((obj.scores[1:]!=obj.scores[:-1])|
                        (obj.image_index[1:]!=obj.image_index[:-1]))+1,len(obj.scores)]
        obj.tie_runs=[(int(a),int(b)) for a,b in zip(boundaries[:-1],boundaries[1:]) if b-a>1]
        obj.reference=obj.accumulate(np.ones(len(obj.ids),dtype=int))
        return obj

    def accumulate(self,counts):
        counts=np.asarray(counts,dtype=np.int64)
        if counts.shape!=self.gt_counts.shape or (counts<0).any(): raise ValueError("Invalid image multiplicities")
        positives=int(counts@self.gt_counts)
        if not positives: return np.full(10,np.nan)
        multiplicity=counts[self.image_index]
        index=np.repeat(np.arange(len(self.scores)),multiplicity)
        if self.tie_runs:
            offsets=np.r_[0,np.cumsum(multiplicity)]
            for a,b in self.tie_runs:
                n=int(multiplicity[a])
                if n>1: index[offsets[a]:offsets[b]]=np.tile(np.arange(a,b),n)
        if not len(index): return np.zeros(10)
        valid=~self.ignored[:,index]; matched=self.matches[:,index]
        tp=np.cumsum(matched&valid,axis=1,dtype=np.float64)
        fp=np.cumsum((~matched)&valid,axis=1,dtype=np.float64)
        recall=tp/positives
        precision=tp/(tp+fp+np.spacing(1))
        precision=np.maximum.accumulate(precision[:,::-1],axis=1)[:,::-1]
        grid=np.linspace(0,1,101); result=[]
        for rec,pr in zip(recall,precision):
            at=np.searchsorted(rec,grid,side="left"); p=np.zeros(101); ok=at<len(pr); p[ok]=pr[at[ok]]; result.append(p.mean())
        return np.asarray(result)


def metrics(vector):
    return {"AP":float(vector.mean()),"AP75":float(vector[5]),"AP80":float(vector[6]),
            "AP90":float(vector[8]),"AP95":float(vector[9]),"AP80_95_mean":float(vector[6:].mean())}
