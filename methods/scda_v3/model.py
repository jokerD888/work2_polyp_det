"""Independent auxiliary head; all native detection losses and eval inherited."""
import torch
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import ImageList
from maskdino.maskdino import MaskDINO

from methods.scda_proposal.model import SCDAProposalMaskDINO
from .aux_head import clone_aux_head
from .config import V3Options
from .assigner import assign
from .loss import auxiliary_loss


@META_ARCH_REGISTRY.register()
class SCDAv3MaskDINO(MaskDINO):
    @configurable
    def __init__(self, *, native_kwargs, v3_options, box_weight, giou_weight):
        super().__init__(**native_kwargs)
        self.v3_options = v3_options
        self.v3_box_weight, self.v3_giou_weight = box_weight, giou_weight
        self.aux_box_head = clone_aux_head(self.sem_seg_head.predictor._bbox_embed)
        self.v3_audit_callback = None

    @classmethod
    def from_config(cls, cfg):
        return {"native_kwargs": super().from_config(cfg), "v3_options": V3Options.from_cfg(cfg),
                "box_weight": cfg.MODEL.MaskDINO.BOX_WEIGHT,
                "giou_weight": cfg.MODEL.MaskDINO.GIOU_WEIGHT}

    # Reuse the proven LSJ valid-pixel/native-target coordinate plumbing.
    _scda_targets = SCDAProposalMaskDINO._scda_targets

    def sync_aux_from_native(self):
        """Once AFTER initialization checkpoint loading, before the first step.

        Prediction/checkpoint reload must never call this, or trained aux state
        would be overwritten. The worker calls it only for a NEW training run.
        """
        self.aux_box_head.load_state_dict(self.sem_seg_head.predictor._bbox_embed.state_dict(), strict=True)

    def forward(self, batched_inputs):
        if not self.training or not self.v3_options.enabled:
            return super().forward(batched_inputs)
        images = [record["image"].to(self.device) for record in batched_inputs]
        images = [(image - self.pixel_mean) / self.pixel_std for image in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        features = self.backbone(images.tensor)
        if not all("instances" in record for record in batched_inputs):
            raise ValueError("V3 training requires explicit Instances, including empty GT")
        instances = [record["instances"].to(self.device) for record in batched_inputs]
        targets = self.prepare_targets(instances, images)
        outputs, mask_dict = self.sem_seg_head(features, targets=targets)
        payload = outputs.pop("scda_payload")
        losses = self.criterion(outputs, targets, mask_dict)
        for key in list(losses):
            if key in self.criterion.weight_dict:
                losses[key] *= self.criterion.weight_dict[key]
            else:
                losses.pop(key)
        auxiliary_targets = self._scda_targets(batched_inputs, instances, targets, images)
        assignments = assign(payload, auxiliary_targets, self.v3_options)
        auxiliary, counts = auxiliary_loss(self.aux_box_head, payload, auxiliary_targets, assignments,
                                           self.v3_options, self.v3_box_weight, self.v3_giou_weight)
        if self.v3_audit_callback is not None:
            self.v3_audit_callback(self, losses, auxiliary, payload, auxiliary_targets, assignments)
        if self.v3_options.diagnostics:
            from .diagnostics import record_audit
            record_audit(assignments, counts, auxiliary)
        losses.update(auxiliary)
        return losses
