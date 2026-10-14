"""Authorized real-model precheck. Never initializes CUDA before baseline exits."""
import argparse
import copy
import dataclasses
import datetime as dt
import gc
import os
from pathlib import Path
import random
import sys
import traceback

import numpy as np
import torch

from polypbench.io import read_json, write_json, sha256
from .experiment import load_experiment


def build_cfg(experiment, enabled, output):
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    import train_net
    from .config import add_scda_config
    from .integration import configure_model
    cfg = get_cfg()
    add_deeplab_config(cfg)
    train_net.add_maskdino_config(cfg)
    add_scda_config(cfg)
    cfg.merge_from_file(experiment["model"]["config"])
    for key, value in experiment["overrides"].items():
        cfg.merge_from_list([key, repr(value)])
    for key, value in experiment["scda"].items():
        setattr(cfg.MODEL.SCDA, key.upper(), value)
    cfg.MODEL.SCDA.ENABLED = enabled
    cfg.MODEL.SCDA.DIAGNOSTICS = True
    configure_model(cfg)
    cfg.MODEL.DEVICE = "cpu"  # build/count on CPU, move only one model at a time
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = experiment["model"]["num_classes"]
    cfg.MODEL.WEIGHTS = experiment["model"]["weights"]
    cfg.DATASETS.TRAIN = ("bench_train",)
    cfg.DATASETS.TEST = ("bench_val",)
    cfg.SEED = experiment["seed"]
    cfg.OUTPUT_DIR = str(output)
    cfg.TEST.AUG.ENABLED = False
    cfg.TEST.DETECTIONS_PER_IMAGE = experiment["model"].get("detections_per_image", 100)
    cfg.freeze()
    return cfg


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    torch.cuda.set_rng_state_all(state[3])


def train_snapshot(model, batch, amp):
    model.train()
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", enabled=amp):
        losses = model(batch)
    total = sum(losses.values())
    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite native/off loss")
    total.backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"Non-finite gradient: {name}")
            gradients[name] = parameter.grad.detach().cpu().clone()
    return {"losses": {name: value.detach().float().cpu() for name, value in losses.items()}, "gradients": gradients}


def compare_snapshots(reference, actual, amp, evidence=None, enforce=True):
    if reference["losses"].keys() != actual["losses"].keys() or reference["gradients"].keys() != actual["gradients"].keys():
        raise AssertionError("Native/off loss or gradient keys changed")
    loss_max, grad_max = 0., 0.
    grad_rtol, grad_atol = (5e-3, 5e-4) if amp else (1e-4, 1e-5)
    for key, value in reference["losses"].items():
        torch.testing.assert_close(value, actual["losses"][key], rtol=1e-6, atol=1e-6, msg=key)
        loss_max = max(loss_max, float((value-actual["losses"][key]).abs().max()))
    differences, failed = {}, []
    for key, value in reference["gradients"].items():
        difference = value.double()-actual["gradients"][key].double()
        maximum = float(difference.abs().max())
        norm = float(value.double().norm())
        difference_norm = float(difference.norm())
        differences[key] = {"max_abs": maximum, "reference_norm": norm,
                            "relative_l2": difference_norm / norm if norm else difference_norm,
                            "elementwise_close": bool(torch.allclose(value, actual["gradients"][key], rtol=grad_rtol, atol=grad_atol)),
                            "close": difference_norm <= grad_atol + grad_rtol * norm}
        if not differences[key]["close"]:
            failed.append(key)
        grad_max = max(grad_max, maximum)
    result = {"passed": not failed, "loss_max_abs_difference": loss_max, "gradient_max_abs_difference": grad_max,
            "gradient_rtol": grad_rtol, "gradient_atol": grad_atol,
            "failed_parameter_count": len(failed),
            "elementwise_failed_parameter_count": sum(not value["elementwise_close"] for value in differences.values()),
            "per_parameter": differences,
            "note": "Per-parameter L2 error uses the same fixed rtol/atol budget; elementwise cancellation around zero fails even native-native CUDA repeats. Native-native is also verified. Losses and eval remain separately compared."}
    if evidence is not None:
        write_json(evidence, result)
    if failed and enforce:
        raise AssertionError(f"Gradient comparison failed for {len(failed)} parameters; first={failed[0]}; evidence={evidence}")
    return result


@torch.no_grad()
def eval_snapshot(model, batch, amp=False):
    model.eval()
    with torch.autocast("cuda", enabled=amp):
        predictions = model(batch)
    return [{name: (field.tensor if hasattr(field, "tensor") else field).detach().cpu().clone()
             for name, field in prediction["instances"].get_fields().items()} for prediction in predictions]


def run_precheck(experiment, output, batches=32):
    from detectron2.data import DatasetCatalog, MetadataCatalog, DatasetMapper
    from detectron2.data.datasets import load_coco_json
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.utils.events import EventStorage
    from detectron2.utils.file_io import PathManager
    from .integration import register
    from .mapper import SCDALSJMapper
    from polypbench.adapters.bitmask_mapper import BitmaskLSJMapper
    from .diagnostics import parameter_audit, gradient_audit, proposal_diagnostics
    from .audit_summary import summarize_decoupling
    import train_net
    register(experiment["repo"]["path"])
    for split in ("train", "val"):
        entry = experiment["dataset"][split]
        name = f"bench_{split}"
        DatasetCatalog.register(name, lambda e=entry, n=name: load_coco_json(
            e["annotations"], e["image_root"], n, extra_annotation_keys=["id", "scda_reliable"]))
        MetadataCatalog.get(name).set(json_file=entry["annotations"], image_root=entry["image_root"], evaluator_type="coco")
    cfg_a, cfg_f = build_cfg(experiment, False, output), build_cfg(experiment, True, output)
    (output / "resolved_A.yaml").write_text(cfg_a.dump(), encoding="utf-8")
    (output / "resolved_F.yaml").write_text(cfg_f.dump(), encoding="utf-8")
    # Verification-only numerical controls, not a change to either training
    # recipe. CuDNN's default TF32/nondeterministic backward was observed to
    # fail elementwise checks even for repeated native A with identical RNG.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {"status": "running", "seed": experiment["seed"], "checks": {},
              "verification_math": {"cudnn_deterministic": True, "tf32": False,
                                    "training_worker_unchanged": True}}
    def save(phase):
        report["phase"] = phase
        report["updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(output / "report.json", report)
    save("cpu_model_build")
    random.seed(experiment["seed"])
    np.random.seed(experiment["seed"])
    torch.manual_seed(experiment["seed"])
    a, f = build_model(cfg_a), build_model(cfg_f)
    audit_a = parameter_audit(a, train_net.Trainer.build_optimizer(cfg_a, a))
    audit_f = parameter_audit(f, train_net.Trainer.build_optimizer(cfg_f, f))
    write_json(output / "parameters_A.json", audit_a)
    write_json(output / "parameters_F.json", audit_f)
    if audit_a != audit_f:
        raise AssertionError("A/F parameters, frozen state, shapes or optimizer groups differ")
    report["checks"]["parameter_optimizer_alignment"] = {"passed": True, "unique_parameters": audit_a["unique_parameters"],
                                                          "trainable_parameters": audit_a["unique_trainable_parameters"]}
    DetectionCheckpointer(a).load(cfg_a.MODEL.WEIGHTS)
    f.load_state_dict(a.state_dict(), strict=True)
    report["initialization"] = {"source": cfg_a.MODEL.WEIGHTS, "sha256": sha256(PathManager.get_local_path(cfg_a.MODEL.WEIGHTS)),
                                "note": "Same ImageNet initial weights; not the trained baseline checkpoint."}
    records = DatasetCatalog.get("bench_train")
    raw_counts = [len([ann for ann in record["annotations"] if not ann.get("iscrowd", 0)]) for record in records]
    report["train_dataset"] = {"images": len(records), "boxes": sum(raw_counts), "single_box_images": raw_counts.count(1),
                                "multi_box_images": sum(count > 1 for count in raw_counts)}
    random.seed(experiment["seed"]); np.random.seed(experiment["seed"]); torch.manual_seed(experiment["seed"])
    mapper = SCDALSJMapper(cfg_f, "single_instance")
    base_mapper = BitmaskLSJMapper(cfg_a, "annotations")
    order = torch.randperm(len(records), generator=torch.Generator().manual_seed(experiment["seed"])).tolist()
    mapped = []
    for index in order[:batches * 4]:
        state = rng_state()
        native = base_mapper(records[index])
        restore_rng(state)
        modified = mapper(records[index])
        for key in ("image", "padding_mask"):
            torch.testing.assert_close(native[key], modified[key], rtol=0, atol=0)
        for key in ("gt_masks", "gt_classes"):
            torch.testing.assert_close(getattr(native["instances"], key), getattr(modified["instances"], key), rtol=0, atol=0)
        torch.testing.assert_close(native["instances"].gt_boxes.tensor, modified["instances"].gt_boxes.tensor, rtol=0, atol=0)
        if modified["scda_original_gt_count"] > 1 and modified["instances"].gt_scda_reliable.any():
            raise AssertionError("Multi-box image incorrectly accepted after augmentation")
        mapped.append(modified)
    report["checks"]["real_mapper_alignment"] = {"passed": True, "images": len(mapped), "comparison": "exact image/padding/bbox/mask/class"}
    # Use source-reliable foreground examples for gradient inspection, keeping
    # the random representative sample untouched for candidate availability.
    good = [record for record in mapped if record["instances"].gt_scda_reliable.any() and record["instances"].gt_masks.any()]
    if len(good) < 4:
        raise RuntimeError("No reliable four-image batch for gradient validation")
    batch = good[:4]
    val_record = DatasetCatalog.get("bench_val")[0]
    eval_batch = [DatasetMapper(cfg_a, False)(val_record)]
    references = {}
    a.cuda()
    save("native_reference")
    eval_a = {amp: eval_snapshot(a, eval_batch, amp) for amp in (False, True)}
    for amp in (False, True):
        references[amp] = {"rng": rng_state(), "values": train_snapshot(a, batch if amp else batch[:1], amp)}
        restore_rng(references[amp]["rng"])
        repeated = train_snapshot(a, batch if amp else batch[:1], amp)
        report["checks"][f"native_repeat_{amp}"] = compare_snapshots(references[amp]["values"], repeated, amp,
            output / f"native_repeat_{amp}.json")
        del repeated
    a.zero_grad(set_to_none=True); a.cpu()
    gc.collect(); torch.cuda.empty_cache()
    f.cuda()
    options = f.scda_options
    f.scda_options = f.sem_seg_head.predictor.scda_options = dataclasses.replace(options, enabled=False)
    save("scda_off_alignment")
    for amp in (False, True):
        restore_rng(references[amp]["rng"])
        actual = train_snapshot(f, batch if amp else batch[:1], amp)
        report["checks"][f"off_alignment_{'amp_b4' if amp else 'fp32_b1'}"] = compare_snapshots(references[amp]["values"], actual, amp,
            output / f"off_alignment_{amp}.json")
        del actual
    del references
    f.zero_grad(set_to_none=True)
    f.scda_options = f.sem_seg_head.predictor.scda_options = options
    for amp in (False, True):
        eval_f = eval_snapshot(f, eval_batch, amp)
        for left, right in zip(eval_a[amp], eval_f):
            if left.keys() != right.keys(): raise AssertionError("Eval field keys changed")
            for key in left:
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0, msg=key)
    report["checks"]["enabled_eval_equivalence"] = {"passed": True, "tolerance": "exact", "precisions": ["fp32", "amp"], "fields": list(eval_a[False][0])}
    save("enabled_gradient_health")
    images, proposals = [], []
    health = {}
    def callback(model, native_losses, auxiliary, payload, targets, assignments):
        if health.get("collect_gradients"):
            health["gradient_report"] = gradient_audit(model, native_losses, auxiliary, payload)
        images.extend(assignment.audit for assignment in assignments)
        proposals.extend(proposal_diagnostics(payload, targets, assignments))
    f.scda_audit_callback = callback
    f.train()
    with EventStorage(0) as storage:
        storage.scda_output_dir = str(output)
        for amp in (False, True):
            health["collect_gradients"] = True
            f.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp):
                losses = f(batch if amp else batch[:1])
            gradient_report = health["gradient_report"]
            write_json(output / f"gradients_{'amp' if amp else 'fp32'}.json", gradient_report)
            for head in ("class_embed", "bbox_embed"):
                for name in ("cls_weighted", "box_weighted"):
                    value = gradient_report["parameter_space"][head]["gradient_norms"][name]
                    if not np.isfinite(value): raise FloatingPointError("Non-finite auxiliary gradient")
                relevant = "cls_weighted" if head == "class_embed" else "box_weighted"
                if gradient_report["parameter_space"][head]["gradient_norms"][relevant] <= 0:
                    raise AssertionError(f"No auxiliary gradient to shared {head}")
            total = sum(losses.values())
            if not torch.isfinite(total): raise FloatingPointError("Non-finite enabled loss")
            total.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in f.parameters()):
                raise FloatingPointError("Enabled total gradient is non-finite")
            report["checks"][f"enabled_gradient_{'amp' if amp else 'fp32'}"] = {"passed": True, "losses": {name: float(value.detach()) for name, value in losses.items() if name.startswith("loss_scda")}}
            del losses, total
        f.zero_grad(set_to_none=True)
        health["collect_gradients"] = False
        images.clear(); proposals.clear()
        save("representative_train_candidate_statistics")
        with torch.no_grad():
            for start in range(0, len(mapped), 4):
                with torch.autocast("cuda", enabled=True):
                    f(mapped[start:start+4])
                storage.step()
    summary = summarize_decoupling(images)
    summary["seed"] = experiment["seed"]
    summary["parameter_stage"] = "ImageNet initialization, no optimizer update"
    summary["proposal_diagnostics"] = proposals
    write_json(output / "SCDA_DECOUPLING_EFFECTIVENESS.json", summary)
    counts = summary["nonempty_valid_reliable_gt"]
    if counts["cls_subset_reg_violations"] or not counts["cls_positive_count"] or not counts["reg_positive_count"]:
        raise AssertionError("Candidate validation failed: no auxiliary supervision or nested-set violation")
    report["checks"]["candidate_availability"] = {"passed": True, **{key: value for key, value in summary.items() if key not in ("per_image", "proposal_diagnostics")}}
    report["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated()
    report["status"] = "passed"
    report["decision"] = "Hard engineering checks passed; short smoke remains required. No research performance GO claimed."
    save("done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="methods/scda_proposal/configs/kvasir.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batches", type=int, default=32)
    args = parser.parse_args()
    experiment = load_experiment(args.config)
    root = Path(experiment["project_root"])
    # Check durable phase markers before importing/building native GPU modules.
    baseline = root / "runs" / "maskdino-official-s42"
    if read_json(baseline / "train" / "state.json")["status"] != "completed" or read_json(baseline / "predict_val" / "state.json")["status"] != "completed":
        raise RuntimeError("Wait until baseline training AND prediction complete")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    torch.set_num_threads(2)
    sys.path.insert(0, experiment["repo"]["path"])
    try:
        import train_net  # makes native registration available
        run_precheck(experiment, output, args.batches)
    except BaseException as error:
        failure = {"status": "failed", "error": str(error), "traceback": traceback.format_exc(),
                   "updated_at": dt.datetime.now(dt.timezone.utc).isoformat()}
        write_json(output / "failure.json", failure)
        previous = read_json(output / "report.json") if (output / "report.json").exists() else {}
        write_json(output / "report.json", {**previous, **failure})
        raise


if __name__ == "__main__":
    main()
