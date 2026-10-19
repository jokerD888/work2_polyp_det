"""Native model/criterion subclasses: add one final-layer loss, reuse matching."""
from dataclasses import asdict

from detectron2.modeling import META_ARCH_REGISTRY
from maskdino.maskdino import MaskDINO
from maskdino.modeling.criterion import SetCriterion

from .config import MSSLOptions
from .loss import matched_loss


class MSSLCriterion(SetCriterion):
    def forward(self, outputs, targets, mask_dict=None):
        # Native forward computes final matching once. get_loss sees that same
        # outputs object only for final losses; encoder/aux/DN are distinct dicts.
        self._final_outputs = outputs
        self._extra_loss = None
        try:
            losses = super().forward(outputs, targets, mask_dict)
            if self._extra_loss is None:
                raise RuntimeError("Native final box-loss hook was not called")
            losses["loss_mssl"] = self._extra_loss
            return losses
        finally:
            self._final_outputs = None
            self._extra_loss = None

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        result = super().get_loss(loss, outputs, targets, indices, num_masks)
        if loss == "boxes" and outputs is self._final_outputs:
            self._extra_loss, self.last_diagnostics = matched_loss(outputs, targets, indices, self.options)
            callback = getattr(self, "probe_callback", None)
            if callback is not None:
                callback(self, outputs, targets, indices, result)
        return result


@META_ARCH_REGISTRY.register()
class MSSLMaskDINO(MaskDINO):
    @classmethod
    def from_config(cls, cfg):
        params = super().from_config(cfg)
        options = MSSLOptions(**{k: getattr(cfg.MODEL.MSSL, k.upper())
                                 for k in asdict(MSSLOptions())}).validate()
        if options.enabled:
            native = params["criterion"]
            keys = ("num_classes", "matcher", "weight_dict", "eos_coef", "losses", "num_points",
                    "oversample_ratio", "importance_sample_ratio", "dn", "dn_losses", "panoptic_on",
                    "semantic_ce_loss")
            criterion = MSSLCriterion(**{k: getattr(native, k) for k in keys})
            criterion.options = options
            # Added after native construction: never replicated to decoder/DN/encoder.
            criterion.weight_dict = dict(criterion.weight_dict, loss_mssl=options.weight)
            params["criterion"] = criterion
        return params

    def prepare_targets(self, targets, images):
        result = super().prepare_targets(targets, images)
        for target, instances in zip(result, targets):
            target["mssl_eligible"] = instances.gt_mssl_eligible
        return result


def register():
    """Import-time registry entry; callable for explicit worker initialization."""
    return MSSLMaskDINO
