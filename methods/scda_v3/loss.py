"""Same per-GT L1/GIoU objective and global-GT normalization; no classification."""
import torch
import torch.distributed as dist

from methods.scda_proposal.scda_loss import _per_gt_sum, aligned_giou, cxcywh_to_xyxy
from .aux_head import predict_selected


def auxiliary_loss(head, payload, targets, assignments, options, box_weight=5., giou_weight=2.):
    if len(targets) != len(assignments) or len(targets) != payload["output_memory"].shape[0]:
        raise ValueError("V3 auxiliary batch size mismatch")
    # A zero auxiliary weight must not add even a zero-valued backward branch
    # to shared features: that can reorder floating-point gradient reductions.
    # Aux parameters still participate with finite zero gradients for DDP.
    prediction_payload = payload if options.lambda_aux else dict(
        payload, output_memory=payload["output_memory"].detach())
    numerator = prediction_payload["output_memory"].reshape(-1)[:0].float().sum()
    count = 0
    for b, (target, assignment) in enumerate(zip(targets, assignments)):
        p, j = assignment.reg.token_indices, assignment.reg.gt_indices
        prediction = predict_selected(head, prediction_payload, b, p)
        # Empty sums keep an attached zero for every aux parameter and memory.
        numerator = numerator + prediction.reshape(-1)[:0].float().sum()
        if p.numel():
            with torch.autocast(device_type=prediction.device.type, enabled=False):
                prediction = prediction.float() * assignment.box_scale[p]
                truth = target.boxes[j].float()
                if not torch.isfinite(truth).all():
                    raise FloatingPointError("Non-finite selected GT box")
                values = (box_weight * (prediction - truth).abs().sum(-1) +
                          giou_weight * (1 - aligned_giou(cxcywh_to_xyxy(prediction),
                                                         cxcywh_to_xyxy(truth))))
                value, gt_count = _per_gt_sum(values, j)
                numerator = numerator + value
                count += gt_count
    counts = numerator.new_tensor([count], dtype=torch.float32)
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    loss = options.lambda_aux * numerator / (counts.clamp(min=1) / world_size)[0]
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite V3 auxiliary loss")
    return {"loss_v3_box": loss}, counts.detach()
