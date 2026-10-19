"""Real-data GPU acceptance; one disposable optimizer step, never a result run."""
import argparse
import copy
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch

from polypbench.io import write_json
from .config import load_experiment, build_cfg


def rng_state():
    return (random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state())


def restore(state):
    random.setstate(state[0]);np.random.set_state(state[1])
    torch.set_rng_state(state[2]);torch.cuda.set_rng_state(state[3])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    experiment=load_experiment(args.config)
    sys.path.insert(0,experiment["repo"]["path"])
    import train_net
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.data import DatasetMapper
    from detectron2.data.datasets import load_coco_json
    from detectron2.modeling import build_model
    from maskdino.modeling.criterion import SetCriterion
    from maskdino.maskdino import MaskDINO
    from .model import MSSLMaskDINO
    from .data import load_records,MSSLMapper
    from polypbench.adapters.bitmask_mapper import BitmaskLSJMapper
    cfg=build_cfg(experiment,out)
    (out/"resolved.yaml").write_text(cfg.dump(),encoding="utf-8")
    report={"status":"running","checks":{},"gpu":torch.cuda.get_device_name(),
            "scope":"real raw-mask mapper, native loss parity, MSSL gradient, full GPU backward and image-only inference"}
    write_json(out/"report.json",report)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark=False
    try:
        records=load_records(experiment["dataset"]["train"]["annotations"],
            experiment["dataset"]["train"]["image_root"],"mssl_acceptance_train",
            experiment["dataset"]["raw_mask_root"])
        mapper=MSSLMapper(cfg,"annotations")
        plain=BitmaskLSJMapper(cfg,"annotations")
        batch=None
        checked=0
        # Include real multi-box originals to verify that cropping cannot relabel
        # them as reliable single-instance supervision.
        selected=[r for r in records if not r["mssl_source_single"]][:3]+records[:12]
        for record in selected:
            state=rng_state();mapped=mapper(record);restore(state);base=plain(record)
            for key in ("image","padding_mask"):
                torch.testing.assert_close(mapped[key],base[key],rtol=0,atol=0)
            for key in ("gt_masks","gt_classes"):
                torch.testing.assert_close(getattr(mapped["instances"],key),getattr(base["instances"],key),rtol=0,atol=0)
            torch.testing.assert_close(mapped["instances"].gt_boxes.tensor,base["instances"].gt_boxes.tensor,rtol=0,atol=0)
            if not record["mssl_source_single"]:
                assert not mapped["instances"].gt_mssl_eligible.any()
            if record["mssl_source_single"] and mapped["instances"].gt_masks.any():
                batch=batch or [mapped]
            checked+=1
        assert batch is not None
        report["checks"]["shared_mapper_and_original_multi_box_policy"]={"passed":True,"images":checked}
        assert MSSLMaskDINO.forward is MaskDINO.forward
        report["checks"]["native_forward_and_inference_inherited"]=True
        model=build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
        model.train()
        captured={}
        def capture(module,inputs):
            captured["inputs"]=inputs
            captured["rng"]=rng_state()
        handle=model.criterion.register_forward_pre_hook(capture)
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast(device_type="cuda",dtype=torch.float16):
            losses=model(batch)
        handle.remove()
        state_after=rng_state()
        assert "loss_mssl" in losses and torch.isfinite(losses["loss_mssl"])
        assert model.criterion.last_diagnostics["eligible"]>0
        criterion=model.criterion
        keys=("num_classes","matcher","weight_dict","eos_coef","losses","num_points",
              "oversample_ratio","importance_sample_ratio","dn","dn_losses","panoptic_on","semantic_ce_loss")
        native=SetCriterion(**{k:getattr(criterion,k) for k in keys}).cuda()
        native.weight_dict={k:v for k,v in criterion.weight_dict.items() if k!="loss_mssl"}
        restore(captured["rng"])
        with torch.autocast(device_type="cuda",dtype=torch.float16):
            expected=native(*captured["inputs"])
        state_native=rng_state()
        torch.testing.assert_close(state_after[2],state_native[2],rtol=0,atol=0)
        torch.testing.assert_close(state_after[3],state_native[3],rtol=0,atol=0)
        for key in expected:
            if key in native.weight_dict:
                torch.testing.assert_close(losses[key],expected[key]*native.weight_dict[key],rtol=0,atol=0)
        report["checks"]["native_loss_and_rng_parity"]={"passed":True,"native_losses":len(expected)}
        outputs=captured["inputs"][0]
        box_params=[p for n,p in model.named_parameters() if "bbox_embed" in n and n.endswith("layers.2.weight")]
        assert box_params
        grads=torch.autograd.grad(losses["loss_mssl"],[outputs["pred_boxes"]]+box_params,
                                  retain_graph=True,allow_unused=True)
        assert grads[0] is not None and grads[0].abs().sum()>0
        norms=[float(g.float().norm()) for g in grads[1:] if g is not None]
        assert norms and max(norms)>0 and all(np.isfinite(norms))
        report["checks"]["direct_box_and_native_head_gradient"]={"passed":True,"bbox_head_grad_norms":norms,
            "loss_mssl":float(losses["loss_mssl"]),"eligible":criterion.last_diagnostics["eligible"]}
        del expected,outputs,captured
        optimizer=train_net.Trainer.build_optimizer(cfg,model)
        scaler=torch.amp.GradScaler("cuda")
        scaling_attempts=[]
        accumulation_steps=cfg.SOLVER.IMS_PER_BATCH//experiment["runtime"]["micro_batch"]
        for attempt in range(20):
            scaler.scale(sum(losses.values())/accumulation_steps).backward()
            scaler.unscale_(optimizer)
            bad=[name for name,p in model.named_parameters()
                 if p.grad is not None and not torch.isfinite(p.grad).all()]
            scaling_attempts.append({"scale":scaler.get_scale(),"nonfinite_parameters":len(bad),
                                     "examples":bad[:3]})
            # Follow native GradScaler overflow handling, including skipped
            # steps and automatic scale reduction, rather than treating its
            # initial scale as a promise of finite gradients.
            scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
            if not bad:
                break
            del losses
            with torch.autocast(device_type="cuda",dtype=torch.float16):
                losses=model(batch)
            assert all(torch.isfinite(value) for value in losses.values())
        report["amp_scaling_attempts"]=scaling_attempts
        assert not bad,"Gradients remain nonfinite after native automatic loss-scale backoff"
        del losses,optimizer,native,grads
        report["checks"]["full_amp_backward_optimizer_step"]=True
        report["peak_training_allocated_mib"]=torch.cuda.max_memory_allocated()/1024**2
        torch.cuda.empty_cache()
        entry=experiment["dataset"]["val"]
        val=load_coco_json(entry["annotations"],entry["image_root"],"mssl_acceptance_val")
        image=DatasetMapper(cfg,False)(val[0])
        image.pop("instances",None)
        model.eval()
        state_keys=set(model.state_dict())
        with torch.no_grad():
            before=model([image])[0]["instances"]
            # Removing the criterion must not affect image-only inference.
            saved=model.criterion;model.criterion=None
            after=model([image])[0]["instances"]
            model.criterion=saved
        for key in ("pred_boxes","scores","pred_classes","pred_masks"):
            a,b=getattr(before,key),getattr(after,key)
            if key=="pred_boxes":a,b=a.tensor,b.tensor
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        assert set(model.state_dict())==state_keys
        report["checks"]["image_only_inference_unchanged"]=True
        report["status"]="passed"
    except BaseException as error:
        report.update(status="failed",error=repr(error))
        raise
    finally:
        write_json(out/"report.json",report)
        print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
