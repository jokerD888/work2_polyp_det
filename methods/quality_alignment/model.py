from dataclasses import asdict

import torch
from detectron2.modeling import META_ARCH_REGISTRY
from maskdino.modeling.criterion import SetCriterion
from methods.mssl.config import MSSLOptions
from methods.mssl.model import MSSLMaskDINO
from methods.mssl.loss import matched_loss
from .loss import quality_loss


class QualityCriterion(SetCriterion):
    def forward(self, outputs, targets, mask_dict=None):
        self._final_outputs = outputs
        self._q_calls = 0
        self._extra_mssl = None
        self.last_q_stats = {}
        try:
            result = super().forward(outputs, targets, mask_dict)
            if self._q_calls != 1:
                raise RuntimeError(f"Final quality classification called {self._q_calls} times")
            if self.mssl_options.enabled:
                if self._extra_mssl is None:
                    raise RuntimeError("Missing final MSSL route")
                result["loss_mssl"] = self._extra_mssl
            return result
        finally:
            self._final_outputs = None
            self._extra_mssl = None

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        if loss == "labels" and outputs is self._final_outputs:
            self._q_calls += 1
            batch, query = self._get_src_permutation_idx(indices)
            target_boxes = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)])
            classes = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices)])
            parts = quality_loss(outputs["pred_logits"], outputs["pred_boxes"][batch, query],
                                 target_boxes, batch, query, classes, num_masks)
            self.last_q_stats = parts["stats"]
            callback = getattr(self, "probe_callback", None)
            if callback is not None:
                parts["native"] = super().get_loss(loss, outputs, targets, indices, num_masks)["loss_ce"]
                callback(self, outputs, targets, indices, parts)
            return {"loss_ce": parts["loss"]}
        result = super().get_loss(loss, outputs, targets, indices, num_masks)
        if loss == "boxes" and outputs is self._final_outputs and self.mssl_options.enabled:
            self._extra_mssl, self.last_diagnostics = matched_loss(outputs, targets, indices, self.mssl_options)
        return result


@META_ARCH_REGISTRY.register()
class QualityAlignMaskDINO(MSSLMaskDINO):
    @classmethod
    def from_config(cls, cfg):
        params = super().from_config(cfg)
        if not cfg.MODEL.QUALITY_ALIGNMENT_ENABLED:
            return params
        native = params["criterion"]
        if native.panoptic_on or native.semantic_ce_loss:
            raise ValueError("This quality experiment requires native sigmoid instance classification")
        names = ("num_classes", "matcher", "weight_dict", "eos_coef", "losses", "num_points",
                 "oversample_ratio", "importance_sample_ratio", "dn", "dn_losses", "panoptic_on", "semantic_ce_loss")
        criterion = QualityCriterion(**{k: getattr(native, k) for k in names})
        criterion.mssl_options = MSSLOptions(**{k: getattr(cfg.MODEL.MSSL, k.upper()) for k in asdict(MSSLOptions())}).validate()
        params["criterion"] = criterion
        return params
