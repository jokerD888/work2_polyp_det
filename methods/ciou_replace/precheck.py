"""Real augmented source image, all live paths, native matching/RNG parity."""
import argparse
import random
import sys
from pathlib import Path
import numpy as np
import torch
from polypbench.io import read_json, write_json, sha256


def run(config, output, training=None, panel_size=1):
    from methods.mssl.config import load_experiment, build_cfg
    experiment = load_experiment(config) if training is None else read_json(Path(training)/"experiment.json")
    sys.path.insert(0, experiment["repo"]["path"])
    import train_net
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.data import DatasetMapper
    from detectron2.data.datasets import load_coco_json
    from maskdino.modeling.criterion import SetCriterion
    from maskdino.maskdino import MaskDINO
    from methods.mssl.data import load_records, MSSLMapper
    from methods.mssl.precheck import rng_state, restore
    from .model import CIoUReplaceMaskDINO
    from .loss import reference_record
    from .audit import BranchAudit

    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "reference": reference_record(), "checks": {}, "panels": [],
              "no_parameter_updates": True, "gpu": torch.cuda.get_device_name()}
    write_json(out/"report.json", report)
    try:
        cfg = build_cfg(experiment, out)
        (out/"resolved.yaml").write_text(cfg.dump(), encoding="utf-8")
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = False
        model = build_model(cfg)
        if training:
            selected = read_json(Path(training)/"train/selected_checkpoint.json")
            checkpoint = Path(training)/"train/model_best.pth"
            assert sha256(checkpoint) == selected["sha256"]
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"], strict=True)
            report["checkpoint_sha256"] = selected["sha256"]
        else:
            DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
        assert CIoUReplaceMaskDINO.forward is MaskDINO.forward
        # Native SetCriterion retains an empty_weight buffer, not trainable parameters.
        assert not list(model.criterion.parameters())
        report["checks"]["no_new_parameters_native_forward_and_inference"] = True
        model.train()
        records = load_records(experiment["dataset"]["train"]["annotations"], experiment["dataset"]["train"]["image_root"],
                               "ciou_audit_train", experiment["dataset"]["raw_mask_root"])
        # Fixed source-only panel: include original multi-box records too.
        multi = sorted([r for r in records if not r["mssl_source_single"]], key=lambda x: x["image_id"])
        single = sorted([r for r in records if r["mssl_source_single"]], key=lambda x: x["image_id"])
        candidates = [single[0], *multi[:1], *single[1:]]
        mapper = MSSLMapper(cfg, "annotations")
        for i, record in enumerate(candidates[:panel_size]):
            seed = 42000+i
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            batch = [mapper(record)]
            if not len(batch[0]["instances"]):
                raise RuntimeError("Fixed audit augmentation has no objects; inspect protocol rather than silently skip")
            captured = {}; match_observed = []
            def capture(module, inputs):
                captured["inputs"] = inputs; captured["rng"] = rng_state()
            def matches(module, inputs, result):
                match_observed.append([(a.tolist(), b.tolist()) for a, b in result])
            hook = model.criterion.register_forward_pre_hook(capture)
            match_hook = model.criterion.matcher.register_forward_hook(matches)
            audit = BranchAudit(model); model.criterion.audit_callback = audit
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                losses = model(batch)
            after = rng_state()
            hook.remove(); model.criterion.audit_callback = None
            assert all(torch.isfinite(x).all() for x in losses.values())
            assert not any("mssl" in k for k in losses)
            coverage = audit.verify(losses)
            row = {"source_image_id": record["image_id"], "source_single_box": record["mssl_source_single"],
                   "augmented_objects": len(batch[0]["instances"]), "replay_seed": seed, **coverage}
            if training is None:
                criterion = model.criterion
                names = ("num_classes", "matcher", "weight_dict", "eos_coef", "losses", "num_points", "oversample_ratio",
                         "importance_sample_ratio", "dn", "dn_losses", "panoptic_on", "semantic_ce_loss")
                native = SetCriterion(**{k: getattr(criterion, k) for k in names}).cuda()
                assert set(native.state_dict()) == set(criterion.state_dict())
                first_matches = list(match_observed); match_observed.clear()
                restore(captured["rng"])
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    expected = native(*captured["inputs"])
                native_after = rng_state()
                assert first_matches == match_observed
                for a, b in zip(after[2:], native_after[2:]):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                assert set(expected) == set(losses)
                differences = {}
                for key in expected:
                    if not key.startswith("loss_giou"):
                        ref = expected[key]*criterion.weight_dict[key]
                        differences[key] = float((losses[key]-ref).abs().detach())
                        torch.testing.assert_close(losses[key], ref, rtol=2e-6, atol=1e-6, msg=key)
                # CUDA point sampling/reductions can differ by an FP32 ULP on
                # replay; document a second native-only run instead of claiming bit parity.
                restore(captured["rng"])
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    repeated = native(*captured["inputs"])
                row["native_replay_roundoff_weighted"] = {
                    key: float(((repeated[key]-expected[key])*criterion.weight_dict[key]).abs().detach())
                    for key in expected if not key.startswith("loss_giou")}
                row["unchanged_loss_absolute_differences"] = differences
                row["native_other_loss_tolerance"] = {"rtol": 2e-6, "atol": 1e-6}
                row["matching_and_torch_rng_exact_parity"] = True
                row["native_other_losses_within_documented_fp32_tolerance"] = True
                restore(after)
                del native, expected, repeated
            match_hook.remove()
            assert all(p.grad is None for p in model.parameters())
            report["panels"].append(row)
            write_json(out/"report.json", report)
            del captured, losses, batch, audit
            torch.cuda.empty_cache()
        report["checks"]["all_active_weighted_regression_paths_and_head_gradients"] = True
        if training is None:
            entry = experiment["dataset"]["val"]
            val = load_coco_json(entry["annotations"], entry["image_root"], "ciou_audit_val")
            image = DatasetMapper(cfg, False)(val[0]); image.pop("instances", None)
            model.eval()
            with torch.no_grad():
                before = model([image])[0]["instances"]
                criterion = model.criterion; model.criterion = None
                after = model([image])[0]["instances"]
                model.criterion = criterion
            for key in ("pred_boxes", "scores", "pred_classes", "pred_masks"):
                a, b = getattr(before, key), getattr(after, key)
                if key == "pred_boxes": a, b = a.tensor, b.tensor
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            report["checks"]["image_only_inference_criterion_removal_exact_parity"] = True
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error)); raise
    finally:
        write_json(out/"report.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True); p.add_argument("--output", required=True)
    p.add_argument("--training"); p.add_argument("--panel-size", type=int, default=1)
    args = p.parse_args(); run(args.config, args.output, args.training, args.panel_size)
