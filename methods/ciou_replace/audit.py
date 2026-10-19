"""No-update weighted per-branch CIoU/GIoU values and box/head gradients."""
import torch
from maskdino.utils import box_ops
from .model import matched_boxes


class BranchAudit:
    def __init__(self, model):
        self.rows = []
        self.parameters = [(n, p) for n, p in model.named_parameters()
                           if "bbox_embed" in n and p.requires_grad]
        if not self.parameters:
            raise RuntimeError("No original bbox head parameters found")

    def __call__(self, criterion, suffix, outputs, targets, indices, normalizer, result):
        key = "loss_giou" + suffix
        weight = criterion.weight_dict[key]
        if weight != 2.:
            raise RuntimeError(f"Unexpected regression weight: {key}: {weight}")
        src, dst = matched_boxes(criterion, outputs, targets, indices)
        with torch.autocast(device_type=src.device.type, enabled=False):
            a = box_ops.box_cxcywh_to_xyxy(src.float())
            b = box_ops.box_cxcywh_to_xyxy(dst.float())
            giou = (1-torch.diag(box_ops.generalized_box_iou(a, b))).sum()/normalizer
        values = {"ciou": result["loss_giou"]*weight, "native_giou": giou*weight}
        record = {"key": key, "weight": weight, "branch": suffix or "final",
                  "matched_pairs": len(src), "native_normalizer": normalizer,
                  "weighted_l1": float((result["loss_bbox"]*criterion.weight_dict["loss_bbox"+suffix]).detach()),
                  "minimum_predicted_xyxy_size": float((a[:, 2:]-a[:, :2]).min()) if len(a) else None,
                  "minimum_target_xyxy_size": float((b[:, 2:]-b[:, :2]).min()) if len(b) else None,
                  "variants": {}}
        inputs = [outputs["pred_boxes"], *[p for _, p in self.parameters]]
        for label, loss in values.items():
            measurement = None
            for scale in (64., 16., 4., 1.):
                grads = torch.autograd.grad(loss*scale, inputs, retain_graph=True, allow_unused=True)
                if all(g is None or bool(torch.isfinite(g).all()) for g in grads):
                    unscaled = [None if g is None else g.detach().float()/scale for g in grads]
                    norm = sum(float(g.square().sum()) for g in unscaled[1:] if g is not None)**.5
                    measurement = {"weighted_loss": float(loss.detach()), "prediction_gradient_l2": float(unscaled[0].norm()),
                                   "shared_bbox_parameter_gradient_l2": norm, "diagnostic_backward_scale": scale,
                                   "parameters_used": sum(g is not None for g in grads[1:]), "finite": True}
                    break
            if measurement is None:
                raise FloatingPointError(f"Diagnostic gradients remain nonfinite: {key}/{label}")
            record["variants"][label] = measurement
        record["ciou_to_giou_head_gradient_ratio"] = record["variants"]["ciou"]["shared_bbox_parameter_gradient_l2"] / (record["variants"]["native_giou"]["shared_bbox_parameter_gradient_l2"]+1e-12)
        self.rows.append(record)

    def verify(self, weighted_losses):
        expected = {k for k in weighted_losses if k.startswith("loss_giou")}
        actual = [row["key"] for row in self.rows]
        if set(actual) != expected or len(set(actual)) != len(actual):
            raise AssertionError(f"Regression path coverage differs: {set(actual)^expected}")
        for row in self.rows:
            value = float(weighted_losses[row["key"]].detach())
            if abs(value-row["variants"]["ciou"]["weighted_loss"]) > 1e-6:
                raise AssertionError("Native model did not apply recorded regression weight")
            head_norm = row["variants"]["ciou"]["shared_bbox_parameter_gradient_l2"]
            original_norm = row["variants"]["native_giou"]["shared_bbox_parameter_gradient_l2"]
            if row["matched_pairs"] and head_norm <= 0:
                # INITIAL_PRED includes detached mask/noisy-GT reference boxes,
                # before any decoder bbox MLP. Native GIoU is zero to the head too.
                if row["branch"] not in ("_0", "_dn_0") or original_norm != 0:
                    raise AssertionError(f"Unexplained zero bbox parameter gradient: {row['key']}")
                row["head_gradient_status"] = "native detached initialization; BOTH CIoU/GIoU head gradients zero; not a learned regression output"
            else:
                row["head_gradient_status"] = "nonzero native bbox parameter gradient"
        return {"keys_weights_gradients": self.rows, "coverage": "all actual enabled regression loss keys, no duplicates",
                "parameter_names": [n for n, _ in self.parameters],
                "scope": "same outputs/matches, before clipping; diagnostic scales divided in FP32; no .grad writes or parameter updates"}
