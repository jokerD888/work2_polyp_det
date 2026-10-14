"""Independent meta-architecture; native criterion and evaluation are inherited."""
import torch
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import ImageList
from maskdino.maskdino import MaskDINO

from .config import SCDAOptions
from .scda_assigner import assign_scda
from .scda_loss import scda_losses
from .structures import SCDATarget


@META_ARCH_REGISTRY.register()
class SCDAProposalMaskDINO(MaskDINO):
    @configurable
    def __init__(self, *, native_kwargs, scda_options, box_weight, giou_weight):
        super().__init__(**native_kwargs)
        # Plain values only; model/state_dict/optimizer gain no modules or tensors.
        self.scda_options = scda_options
        self.scda_box_weight = box_weight
        self.scda_giou_weight = giou_weight
        self.scda_audit_callback = None

    @classmethod
    def from_config(cls, cfg):
        options = SCDAOptions.from_cfg(cfg)
        if options.enabled:
            if not cfg.MODEL.MaskDINO.TWO_STAGE:
                raise ValueError("F-v1 requires native TWO_STAGE=True; it will not turn it on")
            if cfg.MODEL.MaskDINO.SEMANTIC_CE_LOSS:
                raise ValueError("F-v1 sigmoid positive-only BCE does not support SEMANTIC_CE_LOSS")
            if cfg.INPUT.DATASET_MAPPER_NAME != "coco_instance_lsj":
                raise ValueError("Current method implements only the confirmed LSJ instance mapper")
        return {"native_kwargs": super().from_config(cfg), "scda_options": options,
                "box_weight": cfg.MODEL.MaskDINO.BOX_WEIGHT,
                "giou_weight": cfg.MODEL.MaskDINO.GIOU_WEIGHT}

    def _scda_targets(self, inputs, instances, native_targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        result = []
        for record, instance, native in zip(inputs, instances, native_targets):
            h, w = instance.image_size
            if not instance.has("gt_scda_reliable") or not instance.has("gt_scda_ids"):
                raise ValueError("Enabled SCDA requires SCDALSJMapper reliability metadata")
            if "padding_mask" not in record or tuple(record["padding_mask"].shape) != (h, w):
                raise ValueError("Missing/misaligned transformed LSJ padding mask")
            valid_pixels = torch.zeros((h_pad, w_pad), dtype=torch.bool, device=self.device)
            valid_pixels[:h, :w] = ~record["padding_mask"].to(self.device).bool()
            result.append(SCDATarget(native["labels"], native["boxes"], native["masks"],
                                     instance.gt_scda_reliable, instance.gt_scda_ids,
                                     valid_pixels, (h, w), record.get("scda_source", "unknown"),
                                     record.get("scda_original_gt_count")))
        return result

    def forward(self, batched_inputs):
        if not self.training or not self.scda_options.enabled:
            return super().forward(batched_inputs)
        # This is the native training sequence, with separate auxiliary target
        # metadata. Native targets, matcher, masks, DN and weighting are unchanged.
        images = [record["image"].to(self.device) for record in batched_inputs]
        images = [(image - self.pixel_mean) / self.pixel_std for image in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        features = self.backbone(images.tensor)
        if not all("instances" in record for record in batched_inputs):
            raise ValueError("SCDA training expects explicit Instances, including empty-GT images")
        gt_instances = [record["instances"].to(self.device) for record in batched_inputs]
        targets = self.prepare_targets(gt_instances, images)
        outputs, mask_dict = self.sem_seg_head(features, targets=targets)
        payload = outputs.pop("scda_payload")
        losses = self.criterion(outputs, targets, mask_dict)
        for key in list(losses):
            if key in self.criterion.weight_dict:
                losses[key] *= self.criterion.weight_dict[key]
            else:
                losses.pop(key)
        auxiliary_targets = self._scda_targets(batched_inputs, gt_instances, targets, images)
        assignments = assign_scda(payload, auxiliary_targets, self.scda_options)
        auxiliary = scda_losses(payload, auxiliary_targets, assignments, self.scda_options,
                                self.scda_box_weight, self.scda_giou_weight)
        # Outside the native weight_dict loop: lambda and native box coefficients
        # have already been applied exactly once by scda_losses().
        if self.scda_options.diagnostics:
            from .diagnostics import record_training_audit
            if self.scda_audit_callback is not None:
                # Opt-in offline gradient inspection; never enabled by default.
                self.scda_audit_callback(self, losses, auxiliary, payload,
                                         auxiliary_targets, assignments)
            record_training_audit(payload, auxiliary_targets, assignments, auxiliary)
        losses.update(auxiliary.losses)
        return losses
