"""Explicit audit helpers. None are run by importing this module.

No GO/NO-GO thresholds are invented here. Candidate quantities and gradient
measurements require a later user-authorized precheck; eval stays native.
"""
import json
from pathlib import Path

import torch

from .scda_loss import cxcywh_to_xyxy


def _pairwise_iou(prediction, truth):
    prediction, truth = cxcywh_to_xyxy(prediction.float()), cxcywh_to_xyxy(truth.float())
    intersect = (torch.minimum(prediction[:, None, 2:], truth[None, :, 2:]) -
                 torch.maximum(prediction[:, None, :2], truth[None, :, :2])).clamp(min=0).prod(-1)
    area_p = (prediction[:, 2:] - prediction[:, :2]).clamp(min=0).prod(-1)
    area_t = (truth[:, 2:] - truth[:, :2]).clamp(min=0).prod(-1)
    return intersect / (area_p[:, None] + area_t[None] - intersect).clamp(min=torch.finfo(torch.float32).tiny)


@torch.no_grad()
def proposal_diagnostics(payload, targets, assignments):
    result = []
    for b, (target, assignment) in enumerate(zip(targets, assignments)):
        selected = payload["topk_indices"][b]
        raw = payload["raw_box_sigmoid"][b, selected].float() * assignment.box_scale[selected]
        valid_gt = torch.isfinite(target.boxes).all(-1) & (target.boxes[:, 2:] > 0).all(-1)
        truth = target.boxes[valid_gt]
        q = selected.numel()
        # Native Top-Q is never filtered/re-ranked by this diagnostic. Report
        # invalid candidates separately; they cannot count as a GT coverage.
        valid_raw = (payload["valid_token_mask"][b, selected] &
                     torch.isfinite(raw).all(-1) & (raw[:, 2:] > 0).all(-1))
        iou = _pairwise_iou(raw, truth)
        iou = torch.where(valid_raw[:, None] & torch.isfinite(iou), iou, torch.zeros_like(iou))
        counts = (iou >= 0.3).sum(0)
        matched = (iou >= 0.3).any(1)
        entry = {"n_queries": q, "n_gt": len(truth),
                 "gt_ids": target.gt_ids[valid_gt].tolist(),
                 "invalid_raw_proposals": int((~valid_raw).sum()),
                 "raw_coverage_03_count": int((counts > 0).sum()),
                 "raw_coverage_05_count": int((iou >= 0.5).any(0).sum()),
                 "topq_per_gt_iou03_counts": counts.tolist(),
                 "matched_topq_fraction": float(matched.sum()) / max(q, 1)}
        # Deterministic best-IoU ownership is ONLY for a non-double-counted
        # diagnostic ratio. It does not participate in model/query selection.
        if len(truth):
            ownership = iou.argmax(1)
            owned_counts = torch.bincount(ownership[matched], minlength=len(truth))
            excess = (owned_counts - 1).clamp(min=0).sum()
        else:
            owned_counts = torch.empty(0, dtype=torch.long, device=raw.device)
            excess = raw.new_zeros(())
        entry["best_iou_owned_topq_per_gt_counts"] = owned_counts.tolist()
        entry["redundant_excess_topq_fraction"] = float(excess) / max(q, 1)
        entry["redundancy_definition"] = "unique matched queries minus represented GTs, divided by Q; best-IoU GT ownership"
        if "decoder_init_boxes" in payload:
            if payload.get("init_from_mask"):
                h_pad, w_pad = target.valid_pixels.shape
                h_image, w_image = target.image_size
                init_scale = raw.new_tensor([w_pad / w_image, h_pad / h_image] * 2)
            else:
                init_scale = assignment.box_scale[selected]
            initial = payload["decoder_init_boxes"][b].float() * init_scale
            init_iou = _pairwise_iou(initial, truth)
            valid_init = torch.isfinite(initial).all(-1) & (initial[:, 2:] > 0).all(-1)
            init_iou = torch.where(valid_init[:, None] & torch.isfinite(init_iou), init_iou, torch.zeros_like(init_iou))
            entry.update({"init_coverage_03_count": int((init_iou >= 0.3).any(0).sum()),
                          "init_coverage_05_count": int((init_iou >= 0.5).any(0).sum()),
                          "invalid_init_boxes": int((~valid_init).sum())})
        result.append(entry)
    return result


def record_training_audit(payload, targets, assignments, auxiliary):
    """Every step when DIAGNOSTICS=True; per-rank JSONL, never eval GT access."""
    from detectron2.utils import comm
    from detectron2.utils.events import get_event_storage
    storage = get_event_storage()
    proposal = proposal_diagnostics(payload, targets, assignments)
    per_gt = [entry for image in assignments for entry in image.audit["per_gt"]]
    reliable = [entry for entry in per_gt if entry["reliable"] and entry["valid_box"] and entry["nonempty_mask"]]
    storage.put_scalar("scda/local_reliable_gt", len(reliable))
    storage.put_scalar("scda/global_cls_gt", float(auxiliary.global_gt_counts[0]))
    storage.put_scalar("scda/global_box_gt", float(auxiliary.global_gt_counts[1]))
    for task in ("cls", "reg"):
        count = sum(entry[f"n_{task}_selected"] == 0 for entry in reliable)
        storage.put_scalar(f"scda/local_{task}_empty_fraction", count / max(len(reliable), 1))
    # audit directory set by the dedicated worker. If no directory was explicitly
    # set, scalar reporting remains available without writing to an unknown path.
    output = getattr(storage, "scda_output_dir", None)
    if output is not None:
        path = Path(output) / f"scda_audit.rank{comm.get_rank()}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"iteration": storage.iter, "rank": comm.get_rank(),
                                     "assignment": [image.audit for image in assignments],
                                     "proposal": proposal,
                                     "global_effective_gt_counts": auxiliary.global_gt_counts.tolist()},
                                    ensure_ascii=False, allow_nan=False) + "\n")


def parameter_audit(model, optimizer=None):
    """Call only in an authorized precheck after building baseline/F models."""
    params = {id(parameter): parameter for parameter in model.parameters()}
    result = {"unique_parameters": sum(parameter.numel() for parameter in params.values()),
              "unique_trainable_parameters": sum(parameter.numel() for parameter in params.values() if parameter.requires_grad),
              "requires_grad": {name: parameter.requires_grad for name, parameter in model.named_parameters()},
              "state_dict_shapes": {name: list(value.shape) for name, value in model.state_dict().items()}}
    if optimizer is not None:
        names = {id(parameter): name for name, parameter in model.named_parameters()}
        result["optimizer_groups"] = [
            {"parameters": [names[id(parameter)] for parameter in group["params"]],
             "lr": group.get("lr"), "weight_decay": group.get("weight_decay")}
            for group in optimizer.param_groups
        ]
    return result


def _gradient_comparison(losses, tensors):
    gradients = {name: torch.autograd.grad(loss, tensors, retain_graph=True, allow_unused=True)
                 for name, loss in losses.items()}
    norms = {}
    for name, values in gradients.items():
        norms[name] = sum(float(value.detach().double().square().sum())
                          for value in values if value is not None) ** 0.5
    cosine = {}
    for name in losses:
        if name == "native":
            continue
        dot = sum(float(a.detach().double().mul(b.detach().double()).sum())
                  for a, b in zip(gradients["native"], gradients[name])
                  if a is not None and b is not None)
        denominator = norms["native"] * norms[name]
        cosine[name] = dot / denominator if denominator else None
    ratios = {name: value / norms["native"] if norms["native"] else None
              for name, value in norms.items() if name != "native"}
    return {"gradient_norms": norms, "norm_ratio_vs_native": ratios, "cosine_vs_native": cosine}


def gradient_audit(model, native_losses, auxiliary, payload):
    """Unscaled gradients; does not change .grad or invoke optimizer/backward.

    Attach as a callback only for explicitly authorized representative batches.
    Reports lambda-weighted and unweighted cls/box separately; feature gradients
    are reported in a separate space from parameter gradients.
    """
    losses = {"native": sum(native_losses.values()),
              "cls_unweighted": auxiliary.unweighted["cls"],
              "box_unweighted": auxiliary.unweighted["box"],
              "cls_weighted": auxiliary.losses["loss_scda_cls"],
              "box_weighted": auxiliary.losses["loss_scda_box"]}
    predictor = model.sem_seg_head.predictor
    groups = {"class_embed": predictor.class_embed,
              "bbox_embed": predictor._bbox_embed,
              "encoder": model.sem_seg_head.pixel_decoder.transformer.encoder,
              "backbone": model.backbone}
    result = {"parameter_space": {}, "feature_space": {}}
    for group, module in groups.items():
        # Shared bbox heads may first appear under decoder.bbox_embed rather than
        # _bbox_embed in named_parameters(). Select the actual module by identity.
        tensors = tuple({id(parameter): parameter for parameter in module.parameters()
                         if parameter.requires_grad}.values())
        result["parameter_space"][group] = _gradient_comparison(losses, tensors) if tensors else {"unavailable": True}
    if "output_memory" in payload and payload["output_memory"].requires_grad:
        result["feature_space"]["output_memory"] = _gradient_comparison(losses, (payload["output_memory"],))
    return result
