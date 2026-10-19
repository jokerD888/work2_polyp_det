"""Sparse class-head gradient probes; no RNG draws, optimizer steps or .grad writes."""
import json
from pathlib import Path
import torch


class QualityGradientProbe:
    def __init__(self, model, output, period):
        self.parameters = [p for n, p in model.named_parameters() if "class_embed" in n and p.requires_grad]
        if not self.parameters:
            raise RuntimeError("No native classification head parameters")
        self.output = Path(output)
        self.period = period

    def callback(self, iteration, micro_index):
        if micro_index != 0 or (iteration not in (0, 10) and iteration % self.period):
            return None

        def probe(criterion, outputs, targets, indices, parts):
            values = {}
            weight = criterion.weight_dict["loss_ce"]
            for name in ("loss", "positive", "negative", "native"):
                attempts = []
                # Fresh native classification heads can have large negative
                # losses. A fixed scale64 diagnostic may overflow FP16 even
                # though the underlying gradient and AMP training are healthy.
                # This predetermined measurement-only backoff never changes
                # the training GradScaler, optimizer, loss or parameter .grad.
                for scale in (64., 8., 1., .125, .015625):
                    grads = torch.autograd.grad(parts[name] * weight * scale, self.parameters, retain_graph=True, allow_unused=True)
                    finite = all(g is None or bool(torch.isfinite(g).all()) for g in grads)
                    attempts.append({'scale':scale,'finite':finite})
                    if finite:
                        norms = [g.detach().float().div(scale).square().sum() for g in grads if g is not None]
                        norm = torch.stack(norms).sum().sqrt() if norms else outputs['pred_logits'].new_tensor(0.)
                        if torch.isfinite(norm): break
                else:
                    raise FloatingPointError(f'Nonfinite class-head probe for {name}: {attempts}')
                values[name] = {"weighted_loss": float(parts[name].detach() * weight),
                    "class_head_gradient_norm": float(norm), 'measurement_scale':scale,'attempts':attempts}
            row = {"iteration": iteration, "micro_index": micro_index, "weight": weight,
                   "scope": "first microbatch, weighted, before accumulation/AMP clipping; diagnostic scale64 with predetermined overflow backoff, divided in FP32; precision limited",
                   "target_stats": parts["stats"], "gradients": values}
            self.output.parent.mkdir(parents=True, exist_ok=True)
            with self.output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        return probe
