"""Explicit real-image CUDA acceptance, no optimizer step or training launch."""
import argparse
import copy
from dataclasses import replace
import gc
from pathlib import Path
import random
import sys
import traceback
import numpy as np
import torch

from polypbench.io import write_json, sha256
from .config import build_cfg
from .experiment import load_experiment


def run(experiment, output, samples=128):
    from detectron2.data import DatasetMapper
    from detectron2.data.datasets import load_coco_json
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import build_model
    from detectron2.utils.file_io import PathManager
    from detectron2.utils.events import EventStorage
    from methods.scda_proposal.precheck import (rng_state, restore_rng, train_snapshot,
                                               eval_snapshot, compare_snapshots)
    from methods.scda_proposal.scda_assigner import assign_scda
    from polypbench.adapters.bitmask_mapper import BitmaskLSJMapper
    from .integration import register
    from .mapper import V3Mapper
    from .run import adapter_sources
    register(experiment["repo"]["path"])
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(experiment["seed"]); np.random.seed(experiment["seed"]); torch.manual_seed(experiment["seed"])
    report = {"status": "running", "checks": {}, "sources": adapter_sources(experiment),
              "optimizer_steps": 0, "training_started": False, "samples": samples,
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "verification_math": "cudnn deterministic, TF32 off; same controls/tolerances as previous native precheck"}
    def save(): write_json(output/"report.json", report)
    save()
    try:
        cfg = build_cfg(experiment, output, "cpu")
        (output/"resolved_V3.yaml").write_text(cfg.dump(), encoding="utf-8")
        native_cfg = cfg.clone(); native_cfg.defrost()
        native_cfg.MODEL.META_ARCHITECTURE = "MaskDINO"
        native_cfg.MODEL.MaskDINO.TRANSFORMER_DECODER_NAME = "MaskDINODecoder"
        native_cfg.MODEL.SCDA_V3.ENABLED = False; native_cfg.freeze()
        entry = experiment["dataset"]["train"]
        # Supplying dataset_name is mandatory: it remaps external category IDs
        # into contiguous single-class label 0, just like the formal worker.
        records = load_coco_json(entry["annotations"], entry["image_root"], "v3_precheck_train",
                                 extra_annotation_keys=["id", "scda_reliable"])
        if not 1 <= samples <= len(records): raise ValueError("Invalid real-image sample count")
        mapper, baseline_mapper = V3Mapper(cfg, "single_instance"), BitmaskLSJMapper(cfg, "annotations")
        order = torch.randperm(len(records), generator=torch.Generator().manual_seed(experiment["seed"])).tolist()[:samples]
        mapped = []
        for index in order:
            state = rng_state(); old = baseline_mapper(records[index]); restore_rng(state)
            new = mapper(records[index])
            for key in ("image", "padding_mask"): torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
            for key in ("gt_masks", "gt_classes"):
                torch.testing.assert_close(getattr(old["instances"], key), getattr(new["instances"], key), rtol=0, atol=0)
            torch.testing.assert_close(old["instances"].gt_boxes.tensor, new["instances"].gt_boxes.tensor, rtol=0, atol=0)
            if new["instances"].gt_classes.numel() and (new["instances"].gt_classes != 0).any():
                raise AssertionError("Category mapping did not produce single-class 0")
            if new["scda_original_gt_count"] > 1 and new["instances"].gt_scda_reliable.any():
                raise AssertionError("Multi-box original became reliable")
            mapped.append(new)
        report["checks"]["real_mapper"] = {"passed": True, "images": len(mapped), "category_id": 0}
        good = [r for r in mapped if r["instances"].gt_scda_reliable.any() and r["instances"].gt_masks.any()]
        if not good: raise RuntimeError("No reliable foreground sample for gradient acceptance")
        batch = good[:1]
        val = experiment["dataset"]["val"]
        val_records = load_coco_json(val["annotations"], val["image_root"], "v3_precheck_val")
        eval_batch = [DatasetMapper(native_cfg, False)(val_records[0])]
        torch.manual_seed(42); native = build_model(native_cfg)
        DetectionCheckpointer(native).load(cfg.MODEL.WEIGHTS)
        from methods.scda_proposal.box_gate_calibration import IMAGENET_R50_SHA256
        initialization_hash = sha256(PathManager.get_local_path(cfg.MODEL.WEIGHTS))
        if initialization_hash != IMAGENET_R50_SHA256: raise AssertionError("Unexpected ImageNet initialization")
        report["initialization_sha256"] = initialization_hash
        torch.manual_seed(42); model = build_model(cfg)
        incompatible = model.load_state_dict(native.state_dict(), strict=False)
        if incompatible.unexpected_keys or any(not key.startswith("aux_box_head.") for key in incompatible.missing_keys):
            raise AssertionError("Unexpected state_dict mismatch")
        model.sync_aux_from_native()
        references, evaluations = {}, {}
        native.cuda()
        for amp in (False, True):
            evaluations[amp] = eval_snapshot(native, eval_batch, amp)
            state = rng_state(); first = train_snapshot(native, batch, amp)
            restore_rng(state); repeated = train_snapshot(native, batch, amp)
            report["checks"][f"native_repeat_{amp}"] = compare_snapshots(first, repeated, amp,
                                                       output/f"native_repeat_{amp}.json")
            references[amp] = (state, first)
        native.cpu(); native.zero_grad(set_to_none=True); del native; gc.collect(); torch.cuda.empty_cache()
        model.cuda(); options = model.v3_options
        model.v3_options = replace(options, lambda_aux=0., diagnostics=False)
        for amp in (False, True):
            restore_rng(references[amp][0]); actual = train_snapshot(model, batch, amp)
            self_zero = actual["losses"].pop("loss_v3_box")
            if float(self_zero) != 0: raise AssertionError("Zero lambda produced nonzero loss")
            aux_grads = {key: value for key, value in actual["gradients"].items() if key.startswith("aux_box_head.")}
            if not aux_grads or any(g.any() for g in aux_grads.values()): raise AssertionError("Zero lambda aux gradients")
            actual["gradients"] = {key: value for key, value in actual["gradients"].items() if not key.startswith("aux_box_head.")}
            report["checks"][f"zero_lambda_{amp}"] = compare_snapshots(references[amp][1], actual, amp,
                                                     output/f"zero_lambda_{amp}.json")
            enabled_eval = eval_snapshot(model, eval_batch, amp)
            for a, b in zip(evaluations[amp], enabled_eval):
                for key in a: torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        del references; gc.collect()
        report["checks"]["eval_exact"] = {"passed": True, "precisions": ["FP32", "CUDA AMP"]}
        model.v3_options = replace(options, diagnostics=False)
        checks = {}
        def callback(current, native_losses, auxiliary, payload, targets, assignments):
            count = sum(a.reg.token_indices.numel() for a in assignments)
            if not count: raise AssertionError("No auxiliary candidates in gradient batch")
            if current.v3_options.assignment_policy == "legacy_quality_iou":
                old = assign_scda(payload, targets, current.v3_options.legacy(False))
                for a, b in zip(assignments, old):
                    if not torch.equal(a.reg.token_indices, b.reg.token_indices) or not torch.equal(a.reg.gt_indices, b.reg.gt_indices):
                        raise AssertionError("V3-A legacy assignment mismatch")
            # Initial prediction must equal native raw boxes (before controlled
            # test-only perturbation of the zero-initialized final aux layer).
            from .aux_head import predict_selected
            if checks.get("initial_equivalence", False):
                for b, a in enumerate(assignments):
                    p = a.reg.token_indices
                    torch.testing.assert_close(predict_selected(current.aux_box_head, payload, b, p),
                                               payload["raw_box_sigmoid"][b, p], rtol=0, atol=0)
            named = [(name, p) for name, p in current.named_parameters() if p.requires_grad]
            grads = torch.autograd.grad(auxiliary["loss_v3_box"], [p for _, p in named], allow_unused=True, retain_graph=True)
            norms = {}
            for prefix in ("aux_box_head.", "backbone.", "sem_seg_head.pixel_decoder.", "sem_seg_head.predictor.enc_output"):
                norms[prefix] = sum(float(g.float().norm()) for (name, p), g in zip(named, grads)
                                    if name.startswith(prefix) and g is not None)
                if not checks.get("initial_equivalence", False) and norms[prefix] <= 0:
                    raise AssertionError(f"Auxiliary gradient missing: {prefix}")
            for (name, p), g in zip(named, grads):
                if name.startswith("sem_seg_head.predictor.") and not name.startswith(
                        ("sem_seg_head.predictor.enc_output.", "sem_seg_head.predictor.enc_output_norm.")):
                    if g is not None and g.any(): raise AssertionError(f"Auxiliary directly updates native head/decoder: {name}")
            checks["current"] = {"selected_tokens": count, "gradient_norm_sums": norms,
                                  "assignments": [a.audit for a in assignments]}
        model.v3_audit_callback = callback
        with EventStorage(0):
            model.train(); model.zero_grad(set_to_none=True); checks["initial_equivalence"] = True
            with torch.autocast("cuda", enabled=True): result = model(batch)
            report["checks"]["initial_aux_equivalence"] = {"passed": True, **checks["current"],
                          "note": "Zero native last layer implies zero INITIAL memory/backbone gradient; expected"}
            del result; checks["initial_equivalence"] = False
            # Disposable diagnostic weights only. No optimizer/update/reuse.
            with torch.no_grad(): model.aux_box_head.layers[-1].weight.fill_(.001)
            for policy in ("legacy_quality_iou", "mask_multiscale"):
                model.v3_options = replace(options, assignment_policy=policy, diagnostics=False)
                for amp in (False, True):
                    model.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", enabled=amp): result = model(batch)
                    sum(result.values()).backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise FloatingPointError("Non-finite full-network gradient")
                    report["checks"][f"gradient_{policy}_{amp}"] = {"passed": True, **checks["current"],
                                       "loss_v3_box": float(result["loss_v3_box"].detach()),
                                       "test_only_final_aux_weight": .001}
                    del result
        if report["sources"] != adapter_sources(experiment): raise RuntimeError("Source changed during acceptance")
        report.update(status="passed", peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
                      scope="Full native DN/FP32/AMP real-image batch1 checks; no optimizer steps, no B16 smoke or formal training")
    except BaseException:
        report.update(status="failed", traceback=traceback.format_exc())
        raise
    finally:
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="methods/scda_v3/configs/v3a.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--allow-gpu", action="store_true")
    args = parser.parse_args()
    if not args.allow_gpu: parser.error("GPU precheck needs --allow-gpu and an idle GPU")
    experiment = load_experiment(args.config)
    sys.path.insert(0, experiment["repo"]["path"])
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    report = run(experiment, output, args.samples)
    print({"status": report["status"], "optimizer_steps": report["optimizer_steps"], "output": str(output)})


if __name__ == "__main__": main()
