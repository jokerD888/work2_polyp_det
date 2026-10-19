"""Stable pinned IA_BCE_loss with one-to-one matching and rank_weight=1.

Reference: FelixCaae/AlignDETR 4b125f7fe2ff8c3468cd71763fcf971d09ee38db,
aligndetr/losses/losses.py (Apache-2.0, Copyright 2023 Zhi Cai).
Soft positive BCE, differentiable p^2 negative BCE, fully detached quality
target. This is the author's IA-BCE component, not BMVC Eq.3 Align Loss.
"""
import torch
import torch.nn.functional as F

REFERENCE_COMMIT = "4b125f7fe2ff8c3468cd71763fcf971d09ee38db"
REFERENCE_SHA256 = "e70e64f589586d516221eb19c70536bd97eae34d385686a691e33378b15d0ebd"


def aligned_iou(boxes, target):
    """Normalized native cxcywh to aligned IoU; never adjust coordinates."""
    if boxes.shape != target.shape or boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("Corresponding Nx4 cxcywh required")
    if not torch.isfinite(boxes).all() or not torch.isfinite(target).all():
        raise FloatingPointError("Nonfinite boxes")
    if (boxes[:, 2:] < 0).any() or (target[:, 2:] <= 0).any():
        raise ValueError("Negative prediction or nonpositive target dimensions")
    lo = torch.maximum(boxes[:, :2] - boxes[:, 2:] / 2, target[:, :2] - target[:, 2:] / 2)
    hi = torch.minimum(boxes[:, :2] + boxes[:, 2:] / 2, target[:, :2] + target[:, 2:] / 2)
    intersection = (hi - lo).clamp_min(0).prod(-1)
    union = boxes[:, 2:].prod(-1) + target[:, 2:].prod(-1) - intersection
    return intersection / union


def quality_loss(logits, boxes, target_boxes, batch_index, query_index, classes, normalizer):
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=logits.device.type, enabled=False):
        z = logits.to(dtype)
        if not torch.isfinite(z).all():
            raise FloatingPointError("Nonfinite classification logits")
        n = torch.as_tensor(normalizer, device=z.device, dtype=dtype)
        if n.numel() != 1 or not torch.isfinite(n).all() or n.item() <= 0:
            raise ValueError("Positive native normalizer required")
        mask = torch.zeros_like(z, dtype=torch.bool)
        mask[batch_index, query_index, classes] = True
        p = z.sigmoid()
        with torch.no_grad():
            iou = aligned_iou(boxes.detach().to(dtype), target_boxes.detach().to(dtype))
        # Boolean indexing sorts indices by image/query/class. Keep the target
        # order aligned to the provided Hungarian indices, which need not sort.
        positive_logits = z[batch_index, query_index, classes]
        if int(mask.sum()) != len(classes):
            raise ValueError("Duplicate matched classification indices")
        with torch.no_grad():
            target = (positive_logits.sigmoid().pow(.25) * iou.pow(.75)).clamp_min(.01)
        positive = F.binary_cross_entropy_with_logits(positive_logits, target, reduction="sum") / n
        negative = (p[~mask].square() * F.softplus(z[~mask])).sum() / n
        total = positive + negative
        if not torch.isfinite(total).all():
            raise FloatingPointError("Nonfinite IA-BCE loss")
        with torch.no_grad():
            pp = positive_logits.sigmoid()
            stats = {"positive_count": float(len(target)), "negative_count": float((~mask).sum()),
                     "positive_loss": float(positive), "negative_loss": float(negative),
                     "positive_logit_grad_norm": float(((pp - target) / n).norm()),
                     "native_positive_focal": float((.25 * (1 - pp).square() * F.softplus(-positive_logits)).sum() / n),
                     "native_negative_focal": float((.75 * p[~mask].square() * F.softplus(z[~mask])).sum() / n)}
            for name, values in (("target", target), ("iou", iou), ("positive_probability", pp)):
                for suffix, value in (("mean", values.mean() if len(values) else z.new_tensor(0)),
                                      ("min", values.min() if len(values) else z.new_tensor(0)),
                                      ("max", values.max() if len(values) else z.new_tensor(0))):
                    stats[f"{name}_{suffix}"] = float(value)
            if len(target):
                for q in (.1, .5, .9):
                    stats[f"target_p{int(q*100)}"] = float(torch.quantile(target, q))
                stats["target_floor_fraction"] = float((target == .01).float().mean())
                stats["positive_suppression_fraction"] = float((pp > target).float().mean())
            else:
                stats.update(target_p10=0., target_p50=0., target_p90=0., target_floor_fraction=0., positive_suppression_fraction=0.)
        return {"loss": total, "positive": positive, "negative": negative, "target": target, "iou": iou, "stats": stats}
