import torch
from torch.nn import functional as F


def supervised_contrast(features, foreground_count, temperature):
    """Same-class positives; self excluded from numerator and denominator.

    Average positives per anchor, then anchors within each class, then the two
    classes equally. Variable unique counts do not change class weighting.
    """
    with torch.autocast(device_type=features.device.type, enabled=False):
        z = F.normalize(features.float(), dim=-1, eps=1e-6)
        logits = z @ z.T / temperature
        n = len(z)
        if not 2 <= foreground_count <= n-2:
            raise ValueError('At least two unique points in each class are required')
        self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
        labels = torch.arange(n, device=z.device) < foreground_count
        positive = (labels[:, None] == labels[None, :]) & ~self_mask
        log_denominator = logits.masked_fill(self_mask, -torch.inf).logsumexp(1)
        terms = log_denominator - (logits*positive).sum(1)/positive.sum(1)
        return (terms[:foreground_count].mean()+terms[foreground_count:].mean())/2


def region_loss(grid, rows, temperature):
    flat = grid.flatten(1).T
    losses = []
    for row in rows:
        ids = torch.as_tensor(list(row['fg'])+list(row['bg']), device=grid.device)
        losses.append(supervised_contrast(flat[ids], len(row['fg']), temperature))
    return torch.stack(losses).mean() if losses else grid.sum()*0
