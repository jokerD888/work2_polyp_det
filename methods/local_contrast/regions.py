"""GT-only paired eligibility and sampling. CPU geometry, independent RNG."""
import hashlib
import math
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter


def occupancy(mask, stride=8):
    h, w = mask.shape
    if h % stride or w % stride:
        raise ValueError('Region grid requires stride-aligned padded images')
    return mask.reshape(h//stride, stride, w//stride, stride).mean((1, 3))


def pools(region, valid, forbidden_background, other_boxes, options):
    area = int(region.sum())
    if not area:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    inner = max(1, math.ceil(options.core_fraction*math.sqrt(area)))
    outer = max(1, math.ceil(options.ring_fraction*math.sqrt(area)))
    # Square structuring elements: explicitly a Chebyshev pixel margin.
    core = minimum_filter(region, size=2*inner+1, mode='constant', cval=0)
    ring = maximum_filter(region, size=2*outer+1, mode='constant', cval=0) & ~region
    fg = core & valid & ~other_boxes
    bg = ring & valid & ~forbidden_background
    good = occupancy(valid) >= options.purity
    return (np.flatnonzero((occupancy(fg) >= options.purity) & good),
            np.flatnonzero((occupancy(bg) >= options.purity) & good))


def plan(record, options, sequence=0):
    instances = record['instances']
    h, w = record['image'].shape[-2:]
    # Training is fixed 1024 LSJ; no implicit coordinate rescaling is allowed.
    if h % 8 or w % 8:
        raise ValueError('Expected stride8-aligned training image')
    valid = ~record['padding_mask'].cpu().numpy().astype(bool)
    semantic = record['local_semantic_mask'].cpu().numpy().astype(bool)
    boxes = instances.gt_boxes.tensor.cpu().numpy()
    masks = instances.gt_masks.cpu().numpy().astype(bool)
    reliable = instances.gt_mssl_eligible.cpu().numpy().astype(bool)
    yy, xx = np.ogrid[:h, :w]
    rectangles = [(xx+0.5 >= b[0]) & (xx+0.5 < b[2]) &
                  (yy+0.5 >= b[1]) & (yy+0.5 < b[3]) for b in boxes]
    cover = np.zeros((h, w), dtype=np.int16)
    for r in rectangles:
        cover += r
    forbidden = semantic | (cover > 0)
    stats = dict(total=len(boxes), used=0, mask_used=0, box_fallback=0,
                 unreliable=0, mask_insufficient=0, box_insufficient=0,
                 skipped=0, fg_points=0, bg_points=0, budget_shortfall=0)
    rows = []
    for index, rectangle in enumerate(rectangles):
        other = (cover-rectangle.astype(np.int16)) > 0
        box = pools(rectangle, valid, forbidden, other, options)
        mask = pools(masks[index], valid, forbidden, other, options) if reliable[index] else None
        box_ok = min(map(len, box)) >= options.min_points
        mask_ok = mask is not None and min(map(len, mask)) >= options.min_points
        stats['unreliable'] += int(not reliable[index])
        stats['mask_insufficient'] += int(reliable[index] and not mask_ok)
        stats['box_insufficient'] += int(not box_ok)
        if not box_ok:
            stats['skipped'] += 1
            stats['budget_shortfall'] += 2*options.points
            continue
        paired_mask = bool(mask_ok)
        # L1 and L2 get the same count independently in each class. If Mask
        # cannot support the minimum, BOTH use identical Box pools and counts.
        nf = min(options.points, len(box[0]), len(mask[0])) if paired_mask else min(options.points, len(box[0]))
        nb = min(options.points, len(box[1]), len(mask[1])) if paired_mask else min(options.points, len(box[1]))
        use_mask = options.variant == 'mask_priority' and paired_mask
        selected = mask if use_mask else box
        key = f'{options.seed}:{record["image_id"]}:{sequence}:{index}'
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')
        rng = np.random.default_rng(seed)
        fg = rng.choice(selected[0], nf, replace=False)
        bg = rng.choice(selected[1], nb, replace=False)
        rows.append(dict(instance=index, fg=fg, bg=bg, paired_mask=paired_mask,
                         source='mask' if use_mask else 'box'))
        stats['used'] += 1
        stats['mask_used'] += int(use_mask)
        stats['box_fallback'] += int(not paired_mask)
        stats['fg_points'] += nf
        stats['bg_points'] += nb
        stats['budget_shortfall'] += 2*options.points-nf-nb
    return rows, stats
