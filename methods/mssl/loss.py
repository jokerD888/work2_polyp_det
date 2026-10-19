"""GT-only directional support CDF loss. All coordinate calculations use FP32.

Pixel centres, widths and boxes share the augmented image coordinate system.
No mask-derived coordinate target, learned teacher, or predicted-mask input.
"""
import torch
import torch.distributed as dist


def cxcywh_to_xyxy(box):
    return torch.cat((box[..., :2] - box[..., 2:] / 2,
                      box[..., :2] + box[..., 2:] / 2), dim=-1)


def side_losses(pred_xyxy, target_xyxy, mask, options):
    """Return four scalar losses, or None for empty/degenerate GT support.

    The support is a normalized axial foreground mass times an exponential
    proximity weight exp(-abs(u-e*) / (bandwidth * GT-axis-size)).
    The coordinate e* remains the provided detection annotation.
    """
    pred = pred_xyxy.float()
    target = target_xyxy.detach().float()
    mask = mask.detach().float()
    h, w = mask.shape
    size = target[2:] - target[:2]
    if (size <= 0).any() or mask.sum() <= 0:
        return None
    size = size.clamp_min(options.min_scale_pixels)
    # Projections are derived once from GT; never selected by predicted edges.
    px, py = mask.sum(0), mask.sum(1)
    result = []
    for edge, axis in enumerate((0, 1, 0, 1)):
        count = w if axis == 0 else h
        u = torch.arange(count, device=pred.device, dtype=torch.float32) + .5
        mass = px if axis == 0 else py
        if options.support == "box":
            # Exact cell occupancy for the same continuous GT rectangle.
            lo, hi = target[axis], target[axis + 2]
            mass = (torch.minimum(u + .5, hi) - torch.maximum(u - .5, lo)).clamp(0, 1)
        positive = mass > 0
        if not positive.any():
            return None
        u, mass = u[positive], mass[positive]
        relative_u = (u - target[edge]) / size[axis]
        # Log normalization avoids support underflow for a GT-box/mask margin.
        log_weight = mass.log() - relative_u.abs() / options.direction_bandwidth
        pi = log_weight.softmax(0)
        delta = (pred[edge] - target[edge]) / size[axis]
        f_pred = (pi * torch.sigmoid((delta - relative_u) / options.temperature)).sum()
        f_gt = (pi * torch.sigmoid(-relative_u / options.temperature)).sum()
        if options.loss_form == "static":
            # G(d)=F(e*+s*d). Derivative w.r.t. normalized error d, not pixels.
            # All weights are GT-only, detached; preserve the four-side reduction.
            at_zero = torch.sigmoid(-relative_u / options.temperature)
            slope = (pi * at_zero * (1 - at_zero)).sum() / options.temperature
            result.append(.5 * slope.detach().square() * delta.square())
        else:
            result.append(.5 * (f_pred - f_gt).square())
    return torch.stack(result)


def matched_loss(outputs, targets, indices, options):
    """Mean per matched eligible GT; denominator follows actual eligible count."""
    boxes = outputs["pred_boxes"].float()
    total = boxes.sum() * 0
    count = 0
    all_sides = []
    for image, (source, destination) in enumerate(indices):
        target = targets[image]
        if "mssl_eligible" not in target:
            raise ValueError("MSSL requires original-instance reliability metadata")
        h, w = target["masks"].shape[-2:]
        scale = boxes.new_tensor([w, h, w, h])
        for src, dst in zip(source.tolist(), destination.tolist()):
            if not bool(target["mssl_eligible"][dst]):
                continue
            pred = cxcywh_to_xyxy(boxes[image, src]) * scale
            truth = cxcywh_to_xyxy(target["boxes"][dst].float()) * scale
            sides = side_losses(pred, truth, target["masks"][dst], options)
            if sides is not None:
                total = total + sides.mean()
                all_sides.append(sides.detach())
                count += 1
    denominator = boxes.new_tensor(float(count))
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(denominator)
        denominator /= dist.get_world_size()
    loss = total / denominator.clamp_min(1)
    diagnostics = {"eligible": count,
                   "side_loss": torch.stack(all_sides).mean(0).cpu().tolist() if all_sides else [0.] * 4}
    return loss, diagnostics
