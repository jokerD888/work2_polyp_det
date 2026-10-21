"""Formal opt-in model: attach before optimizer creation, keep native losses."""
from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from methods.mssl.model import MSSLMaskDINO
from .integration import SupportReferenceAttachment


@META_ARCH_REGISTRY.register()
class SupportReferenceMaskDINO(MSSLMaskDINO):
    @configurable
    def __init__(self, *, support_reference_variant, **kwargs):
        super().__init__(**kwargs)
        self.support_reference_variant = support_reference_variant
        self.reference_attachment = None
        if support_reference_variant != 'native':
            self.reference_attachment = SupportReferenceAttachment(
                self, enabled=True, capture=False, variant=support_reference_variant)
            self.reference_attachment.__enter__()

    @classmethod
    def from_config(cls, cfg):
        params = super().from_config(cfg)
        params['support_reference_variant'] = cfg.MODEL.SUPPORT_REFERENCE_VARIANT
        return params
