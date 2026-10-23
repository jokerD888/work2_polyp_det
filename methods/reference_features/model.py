from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from methods.mssl.model import MSSLMaskDINO
from .integration import ReferenceFeaturesAttachment


@META_ARCH_REGISTRY.register()
class ReferenceFeaturesMaskDINO(MSSLMaskDINO):
    @configurable
    def __init__(self, *, reference_features_variant, **kwargs):
        super().__init__(**kwargs)
        self.reference_features_variant = reference_features_variant
        self.reference_attachment = None
        if reference_features_variant != 'native':
            self.reference_attachment = ReferenceFeaturesAttachment(self, reference_features_variant)
            self.reference_attachment.__enter__()

    @classmethod
    def from_config(cls, cfg):
        params = super().from_config(cfg)
        params['reference_features_variant'] = cfg.MODEL.REFERENCE_FEATURES_VARIANT
        return params
