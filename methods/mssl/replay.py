"""Checkpoint gradient replay on a fixed source-training panel; no optimization."""
import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch

from polypbench.io import read_json, sha256, write_json
from .config import build_cfg, MSSLOptions
from .data import load_records, MSSLMapper
from .gradient_probe import GradientProbe
from .loss import cxcywh_to_xyxy


def run(training, output):
    from detectron2.modeling import build_model
    from .model import MSSLMaskDINO
    cfg_dict = copy.deepcopy(read_json(Path(training)/"experiment.json"))
    # A criterion is needed for baseline replay too; no model state parameters change.
    cfg_dict["mssl"] = {**MSSLOptions().__dict__, **cfg_dict["mssl"], "enabled": True}
    out=Path(output); out.mkdir(parents=True,exist_ok=False)
    cfg=build_cfg(cfg_dict,out)
    records=load_records(cfg_dict["dataset"]["train"]["annotations"],cfg_dict["dataset"]["train"]["image_root"],
                         "gradient_replay_train",cfg_dict["dataset"]["raw_mask_root"])
    panel=sorted([r for r in records if r["mssl_source_single"]],key=lambda x:x["image_id"])[:16]
    mapper=MSSLMapper(cfg,"annotations")
    model=build_model(cfg)
    checkpoint=Path(training)/"train/model_best.pth"
    selected=read_json(Path(training)/"train/selected_checkpoint.json")
    assert sha256(checkpoint)==selected["sha256"]
    model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=False)["model"],strict=True)
    model.train()
    probe=GradientProbe(model,out/"gradient_probes.jsonl",1)
    metadata=[]
    for i,record in enumerate(panel):
        seed=42000+i
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        batch=[mapper(record)]
        base_callback=probe.callback(i,0)
        entry={"image_id":record["image_id"],"replay_seed":seed,"eligible_after_augmentation":
               int(batch[0]["instances"].gt_mssl_eligible.sum()), "sides":[]}
        def callback(criterion, outputs, targets, indices, native):
            base_callback(criterion,outputs,targets,indices,native)
            for b,(src,dst) in enumerate(indices):
                t=targets[b]; h,w=t["masks"].shape[-2:]
                scale=outputs["pred_boxes"].new_tensor([w,h,w,h])
                for si,di in zip(src.tolist(),dst.tolist()):
                    if not bool(t["mssl_eligible"][di]): continue
                    pred=(cxcywh_to_xyxy(outputs["pred_boxes"][b,si].float())*scale).detach()
                    truth=cxcywh_to_xyxy(t["boxes"][di].float())*scale
                    size=(truth[2:]-truth[:2]).clamp_min(criterion.options.min_scale_pixels)
                    mask=t["masks"][di].float(); projections=(mask.sum(0),mask.sum(1))
                    for e,axis in enumerate((0,1,0,1)):
                        mass=projections[axis]; pos=mass>0
                        if not pos.any(): continue
                        u=torch.arange(len(mass),device=mass.device,dtype=torch.float32)[pos]+.5
                        rel=(u-truth[e])/size[axis]
                        pi=(mass[pos].log()-rel.abs()/criterion.options.direction_bandwidth).softmax(0)
                        d=(pred[e]-truth[e])/size[axis]; tau=criterion.options.temperature
                        a=torch.sigmoid((d-rel)/tau); z=torch.sigmoid(-rel/tau)
                        slope=float((pi*a*(1-a)).sum()/tau)
                        slope0=float((pi*z*(1-z)).sum()/tau)
                        grad=float((pi*(a-z)).sum())*slope
                        entry["sides"].append({"side":e,"normalized_error":float(d),"Gprime_d":slope,
                            "Gprime_0":slope0,"gradient_wrt_d":grad,
                            "weak_slope_vs_local_1pct":bool(slope<.01*slope0),
                            "nontrivial_error_abs_d_gt_005":bool(abs(float(d))>.05)})
        model.criterion.probe_callback=callback
        with torch.autocast(device_type="cuda",dtype=torch.float16): losses=model(batch)
        assert all(torch.isfinite(x).all() for x in losses.values())
        model.criterion.probe_callback=None
        metadata.append(entry)
        del losses,batch
        model.zero_grad(set_to_none=True)
    write_json(out/"replay.json",{"status":"passed","training":str(training),"checkpoint_sha256":selected["sha256"],
        "scope":"16 fixed original-single training images; fixed LSJ seeds; training-mode criterion/DN; no updates; checkpoint-local evidence only",
        "saturation_definition":"Gprime(d)<0.01*Gprime(0); also inspect abs(d)>0.05 to distinguish near-target zero loss gradient",
        "panel":metadata})


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--training",required=True); parser.add_argument("--output",required=True)
    args=parser.parse_args(); run(args.training,args.output)
