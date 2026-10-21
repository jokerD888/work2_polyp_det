"""Isolated mathematical prototype; NOT registered with any model or runner.

Only pre-Decoder predictions are accepted. No detector, matching, dataset,
checkpoint, training loop or GPU use is required by this module.
"""
import torch
from torch import nn


def _mass_bins(mass, first, last, bins):
    """Integrate piecewise constant pixel mass in equal bbox-relative bins.

    last is inclusive. Unlike interpolating a small mask crop, this retains
    the mass of isolated pixels and of narrow objects. Empty mass stays zero.
    """
    length = mass.shape[-1]
    edges = first[..., None] + (last - first + 1)[..., None] * torch.linspace(
        0, 1, bins + 1, device=mass.device, dtype=mass.dtype)
    edges = edges.clamp(0, length)
    index = edges.floor().long().clamp(max=length)
    prefix = torch.cat((torch.zeros_like(mass[..., :1]), mass.cumsum(-1)), -1)
    integral = prefix.gather(-1, index)
    value = mass.gather(-1, index.clamp(max=length - 1))
    integral = integral + (edges - index) * value
    return integral.diff(dim=-1) / mass.sum(-1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def support_descriptor(mask_logits, encoder_reference_logits,
                       native_mask_reference_logits, valid_pixels, bins=16):
    """Return detached [B,Q,4*bins+16] descriptors and eligible [B,Q].

    All boxes use the actual native normalized cxcywh reference, not a second
    box extraction. Boundary Encoder references are valid descriptor inputs.
    Profiles include native padded foreground; padding is separately flagged.
    Soft profiles weight ONLY the native positive support, not all sigmoid mass.
    """
    if bins < 1 or mask_logits.ndim != 4:
        raise ValueError("Expected positive bins and [B,Q,H,W] masks")
    batch, queries, height, width = mask_logits.shape
    expected = (batch, queries, 4)
    if encoder_reference_logits.shape != expected or native_mask_reference_logits.shape != expected:
        raise ValueError("References must be [B,Q,4]")
    if valid_pixels.shape != (batch, height, width) or valid_pixels.dtype != torch.bool:
        raise ValueError("valid_pixels must be bool [B,H,W] on the mask lattice")
    if not valid_pixels.flatten(1).any(-1).all():
        raise ValueError("Each image must have valid pixels")
    encoder = encoder_reference_logits.float().sigmoid()
    native = native_mask_reference_logits.float().sigmoid()
    if not (torch.isfinite(mask_logits).all() and torch.isfinite(encoder).all()
            and torch.isfinite(native_mask_reference_logits).all()):
        raise ValueError("Nonfinite mask/reference; preserve native Encoder +/-inf only")
    support = mask_logits > 0
    hard = support.float()
    soft = mask_logits.float().sigmoid() * hard
    px, py = hard.sum(-2), hard.sum(-1)
    x = torch.arange(width, device=mask_logits.device)[None, None]
    y = torch.arange(height, device=mask_logits.device)[None, None]
    xmin = torch.where(px > 0, x, width).amin(-1)
    xmax = torch.where(px > 0, x, -1).amax(-1)
    ymin = torch.where(py > 0, y, height).amin(-1)
    ymax = torch.where(py > 0, y, -1).amax(-1)
    count = hard.sum((-2, -1))
    empty = count == 0
    degenerate = (~empty) & ((xmax <= xmin) | (ymax <= ymin))
    profiles = torch.cat((
        _mass_bins(px, xmin, xmax, bins),
        _mass_bins(py, ymin, ymax, bins),
        _mass_bins(soft.sum(-2), xmin, xmax, bins),
        _mass_bins(soft.sum(-1), ymin, ymax, bins)), -1)
    bbox_area = ((xmax - xmin + 1) * (ymax - ymin + 1)).clamp_min(1)
    occupancy = count / bbox_area
    padded = (hard * (~valid_pixels)[:, None]).sum((-2, -1)) / count.clamp_min(1)
    valid_rows = valid_pixels.any(-1).float().mean(-1)[:, None].expand_as(count)
    valid_cols = valid_pixels.any(-2).float().mean(-1)[:, None].expand_as(count)
    scalars = torch.stack((occupancy, count / (height * width), padded,
        empty.float(), degenerate.float(), ((encoder == 0) | (encoder == 1)).any(-1).float(),
        valid_rows, valid_cols), -1)
    features = torch.cat((profiles, encoder, native, scalars), -1)
    return features, ~(empty | degenerate)


class SupportReferencePrototype(nn.Module):
    """Identity-initialized logit residual, not a Box/Mask interpolation gate.

    Detached inputs deliberately block new gradients into mask/Encoder heads.
    Native auxiliary-0 box loss could train this adapter after integration.
    That routing is NOT tested here: this is only an isolated prototype.
    """
    def __init__(self, bins=16, hidden=64):
        super().__init__()
        self.bins = bins
        # Do not shift the detector/data-augmentation RNG stream on construction.
        with torch.random.fork_rng(devices=[]):
            self.head = nn.Sequential(nn.Linear(4 * bins + 16, hidden), nn.ReLU(),
                                      nn.Linear(hidden, 4))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, mask_logits, encoder_reference_logits,
                native_mask_reference_logits, valid_pixels):
        features, eligible = support_descriptor(mask_logits, encoder_reference_logits,
            native_mask_reference_logits, valid_pixels, self.bins)
        delta = self.head(features)
        base = native_mask_reference_logits.detach()
        adjusted = base + delta.to(base.dtype)
        # These cases remain exactly native; no synthetic Encoder fallback.
        return torch.where(eligible[..., None], adjusted, base)
