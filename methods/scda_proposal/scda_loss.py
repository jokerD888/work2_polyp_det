"""Positive-only shared-head losses, per-GT means and global DDP normalization."""
import torch
import torch.distributed as dist
from torch.nn import functional as F

from .structures import LossResult


def connected_zero(tensor):
    # Empty view retains the original autograd path without summing +/-inf or
    # NaN values from invalid native proposals. Backward scatters zero gradients.
    return tensor.reshape(-1)[:0].float().sum()


def positive_cls_loss(logits, options):
    """Positive-term objective on selected tokens, FP32.

    "bce" is the confirmed F-v1/v2 plain softplus(-logit). "focal_positive"
    multiplies the same term by alpha*(1-p)^gamma with the resolved native
    sigmoid_focal_loss constants (criterion.py), so an already-confident token
    is down-weighted exactly the way native positives are.
    """
    values = F.softplus(-logits)
    if options.cls_loss_form == "focal_positive":
        values = values * options.focal_alpha * (1 - torch.sigmoid(logits)) ** options.focal_gamma
    return values


def cxcywh_to_xyxy(boxes):
    center, half = boxes[..., :2], boxes[..., 2:] / 2
    return torch.cat((center - half, center + half), dim=-1)


def aligned_giou(boxes1, boxes2):
    """O(K) GIoU for paired boxes, FP32; no all-pairs NxN allocation."""
    boxes1, boxes2 = boxes1.float(), boxes2.float()
    intersection = (torch.minimum(boxes1[:, 2:], boxes2[:, 2:]) -
                    torch.maximum(boxes1[:, :2], boxes2[:, :2])).clamp(min=0).prod(-1)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp(min=0).prod(-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp(min=0).prod(-1)
    union = area1 + area2 - intersection
    enclosure = (torch.maximum(boxes1[:, 2:], boxes2[:, 2:]) -
                 torch.minimum(boxes1[:, :2], boxes2[:, :2])).clamp(min=0).prod(-1)
    eps = torch.finfo(torch.float32).tiny
    return intersection / union.clamp(min=eps) - (enclosure - union) / enclosure.clamp(min=eps)


def _per_gt_sum(values, gt_indices):
    # GTs with no positives are not in the denominator. Each participating GT
    # contributes one mean, independently for classification and regression.
    groups, inverse, counts = torch.unique(gt_indices, sorted=True, return_inverse=True, return_counts=True)
    sums = values.new_zeros(groups.numel()).scatter_add(0, inverse, values)
    return (sums / counts.to(values.dtype)).sum(), groups.numel()


def scda_losses(payload, targets, assignments, options, box_weight, giou_weight):
    raw_cls, raw_boxes = payload["raw_cls"], payload["raw_box_sigmoid"]
    if len(targets) != raw_cls.shape[0] or len(assignments) != len(targets):
        raise ValueError("SCDA batch size mismatch")
    numerator_cls = connected_zero(raw_cls)
    numerator_box = connected_zero(raw_boxes)
    n_cls = n_box = 0
    # FP32 includes softplus and coordinate conversion, not only GIoU. Casting
    # these attached tensors preserves gradients to the native shared heads.
    with torch.autocast(device_type=raw_cls.device.type, enabled=False):
        for b, (target, assignment) in enumerate(zip(targets, assignments)):
            p, j = assignment.cls.token_indices, assignment.cls.gt_indices
            if p.numel():
                labels = target.labels[j]
                if ((labels < 0) | (labels >= raw_cls.shape[-1])).any():
                    raise ValueError("GT class index outside native sigmoid classifier")
                values = positive_cls_loss(raw_cls[b, p, labels].float(), options)
                contribution, count = _per_gt_sum(values, j)
                numerator_cls = numerator_cls + contribution
                n_cls += count
            p, j = assignment.reg.token_indices, assignment.reg.gt_indices
            if p.numel():
                prediction = raw_boxes[b, p].float() * assignment.box_scale[p]
                truth = target.boxes[j].float()
                l1 = (prediction - truth).abs().sum(-1)
                giou = aligned_giou(cxcywh_to_xyxy(prediction), cxcywh_to_xyxy(truth))
                values = box_weight * l1 + giou_weight * (1 - giou)
                contribution, count = _per_gt_sum(values, j)
                numerator_box = numerator_box + contribution
                n_box += count
        counts = raw_cls.new_tensor([n_cls, n_box], dtype=torch.float32)
        world_size = 1
        # Both counts are reduced in one collective on EVERY rank, including
        # ranks with zero GTs or no auxiliary candidates. DDP averages gradients.
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        denominator = counts.clamp(min=1) / world_size
        cls_loss = numerator_cls / denominator[0]
        box_loss = numerator_box / denominator[1]
    losses = {"loss_scda_cls": options.lambda_cls * cls_loss,
              "loss_scda_box": options.lambda_box * box_loss}
    if not all(torch.isfinite(value).all() for value in losses.values()):
        raise FloatingPointError("Non-finite SCDA loss; stop rather than silently masking it")
    return LossResult(losses, {"cls": cls_loss, "box": box_loss}, counts.detach())
