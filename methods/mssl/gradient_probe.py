"""Read-only, unscaled gradients on the shared native box head; no RNG draws."""
from dataclasses import replace
import json
from pathlib import Path

import torch

from .loss import matched_loss


class GradientProbe:
    def __init__(self, model, output, period=275):
        self.output = Path(output)
        self.period = period
        # named_parameters removes duplicates: the native decoder shares its MLP.
        self.named = [(n, p) for n, p in model.named_parameters()
                      if "bbox_embed" in n and p.requires_grad]
        if not self.named:
            raise RuntimeError("No shared bbox regression parameters found")

    def callback(self, iteration, micro_index):
        if micro_index != 0 or (iteration not in (0, 10) and iteration % self.period):
            return None

        def measure(criterion, outputs, targets, indices, native_boxes):
            params = [p for _, p in self.named]
            native = sum(native_boxes[k] * criterion.weight_dict[k]
                         for k in ("loss_bbox", "loss_giou"))
            variants = {"native_final_l1_giou": native}
            for label, support, form in (("mask_cdf", "mask", "cdf"),
                                         ("box_cdf", "box", "cdf"),
                                         ("mask_static", "mask", "static")):
                opt = replace(criterion.options, support=support, loss_form=form)
                variants[label] = matched_loss(outputs, targets, indices, opt)[0] * opt.weight
            values = {}
            for label, loss in variants.items():
                # Native AMP scales before backward. An unscaled diagnostic
                # backward can underflow on the FP16 head even when the true
                # scaled training contribution is nonzero. Fixed 64 is a
                # measurement scale only; divide in FP32, never touch .grad.
                measurement_scale = 64.
                gradients = torch.autograd.grad(loss * measurement_scale, params,
                                                retain_graph=True, allow_unused=True)
                finite = all(g is None or bool(torch.isfinite(g).all()) for g in gradients)
                norm = float(torch.stack([(g.detach().float()/measurement_scale).square().sum()
                            for g in gradients if g is not None]).sum().sqrt()) if finite else None
                values[label] = {"loss": float(loss.detach()), "gradient_l2": norm,
                                 "finite": finite,
                                 "diagnostic_backward_scale": measurement_scale,
                                 "parameters_used": sum(g is not None for g in gradients)}
            denominator = values["native_final_l1_giou"]["gradient_l2"]
            for label in ("mask_cdf", "box_cdf", "mask_static"):
                numerator = values[label]["gradient_l2"]
                values[label]["ratio_to_native_final_box"] = (
                    numerator / (denominator + 1e-12)
                    if numerator is not None and denominator is not None else None)
            row = {"optimizer_iteration_zero_based": iteration, "microbatch": micro_index,
                   "scope": "same forward/matches; weighted losses; shared bbox head; diagnostic backward scale64 divided in FP32; before clipping; not full-model or accumulated gradients; zeros may still be precision-limited",
                   "eligible": criterion.last_diagnostics["eligible"],
                   "parameter_names": [n for n, _ in self.named], "variants": values}
            self.output.parent.mkdir(parents=True, exist_ok=True)
            with self.output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")

        return measure
