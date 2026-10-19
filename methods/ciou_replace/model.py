"""Only replace native matched-box regression. All routing stays native."""
import torch
import torch.nn.functional as F
from detectron2.modeling import META_ARCH_REGISTRY
from maskdino.modeling.criterion import SetCriterion
from maskdino.utils import box_ops
from methods.mssl.model import MSSLMaskDINO
from .loss import ciou_pairs, reference_record


def matched_boxes(criterion, outputs, targets, indices):
    src = outputs["pred_boxes"][criterion._get_src_permutation_idx(indices)]
    dst = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)], dim=0)
    return src, dst


class CIoUCriterion(SetCriterion):
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        src, dst = matched_boxes(self, outputs, targets, indices)
        return {"loss_bbox": F.l1_loss(src, dst, reduction="none").sum() / num_boxes,
                # Keep native suffix construction/weight filtering compatible.
                "loss_giou": ciou_pairs(box_ops.box_cxcywh_to_xyxy(src.float()),
                                        box_ops.box_cxcywh_to_xyxy(dst.float())).sum() / num_boxes}

    def loss_boxes_panoptic(self, *args, **kwargs):
        raise RuntimeError("This prespecified experiment is instance-only, not panoptic")

    def forward(self, outputs, targets, mask_dict=None):
        callback = getattr(self, "audit_callback", None)
        if callback is None:
            return super().forward(outputs, targets, mask_dict)
        routes = {id(outputs): ""}
        routes.update({id(o): f"_{i}" for i, o in enumerate(outputs.get("aux_outputs", []))})
        if "interm_outputs" in outputs:
            routes[id(outputs["interm_outputs"])] = "_interm"
        if self.dn != "no" and mask_dict is not None:
            known = mask_dict["output_known_lbs_bboxes"]
            routes[id(known)] = "_dn"
            routes.update({id(o): f"_dn_{i}" for i, o in enumerate(known.get("aux_outputs", []))})
        self.audit_routes = routes
        self.audit_seen = set()
        try:
            return super().forward(outputs, targets, mask_dict)
        finally:
            self.audit_routes = None

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        result = super().get_loss(loss, outputs, targets, indices, num_masks)
        callback = getattr(self, "audit_callback", None)
        if callback is not None and loss == "boxes":
            suffix = self.audit_routes[id(outputs)]
            if suffix in self.audit_seen:
                raise RuntimeError(f"Regression branch calculated twice: {suffix}")
            self.audit_seen.add(suffix)
            callback(self, suffix, outputs, targets, indices, num_masks, result)
        return result


@META_ARCH_REGISTRY.register()
class CIoUReplaceMaskDINO(MSSLMaskDINO):
    @classmethod
    def from_config(cls, cfg):
        if cfg.MODEL.MSSL.ENABLED:
            raise ValueError("CIoU-Replace cannot be combined with MSSL")
        params = super().from_config(cfg)
        native = params["criterion"]
        if native.panoptic_on:
            raise ValueError("CIoU protocol requires native instance box losses")
        reference_record()
        names = ("num_classes", "matcher", "weight_dict", "eos_coef", "losses", "num_points",
                 "oversample_ratio", "importance_sample_ratio", "dn", "dn_losses", "panoptic_on",
                 "semantic_ce_loss")
        params["criterion"] = CIoUCriterion(**{k: getattr(native, k) for k in names})
        return params
