"""A derived head inserts one adapter without duplicating the SCDA decoder."""
from detectron2.config import configurable
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from maskdino.modeling.meta_arch.maskdino_head import MaskDINOHead

from .config import DCNv4Options
from .dcnv4_adapter import DCNv4Residual


def feature_strides(pixel_decoder):
    """Mirror the pinned forward_features ordering, including extra levels.

    Do not interpret the name low2high as a stride contract: the pinned code
    returns reversed transformer_in_features, followed by reversed extras.
    """
    base = list(reversed(pixel_decoder.transformer_feature_strides))
    count = pixel_decoder.total_num_feature_levels - len(base)
    if count < 0:
        raise ValueError("Invalid native feature-level count")
    origin = pixel_decoder.transformer_feature_strides[pixel_decoder.low_resolution_index]
    extra = [origin * 2 ** (i + 1) for i in range(count)][::-1]
    return base + extra if pixel_decoder.feature_order == "low2high" else extra + base


@SEM_SEG_HEADS_REGISTRY.register()
class ProposalDCNv4Head(MaskDINOHead):
    @configurable
    def __init__(self, input_shape, *, dcnv4_options, adapter=None, **native_kwargs):
        super().__init__(input_shape, **native_kwargs)
        self.dcnv4_options = dcnv4_options.validate()
        self.dcnv4_adapter = adapter if adapter is not None else DCNv4Residual(dcnv4_options)
        strides = feature_strides(self.pixel_decoder)
        if strides.count(dcnv4_options.feature_stride) != 1:
            raise ValueError(f"Requested stride {dcnv4_options.feature_stride} not unique in {strides}")
        self.dcnv4_level_index = strides.index(dcnv4_options.feature_stride)
        self.dcnv4_feature_strides = tuple(strides)
        self._source_feature = next((key for key, spec in input_shape.items()
                                     if spec.stride == dcnv4_options.feature_stride), None)

    @classmethod
    def from_config(cls, cfg, input_shape):
        result = super().from_config(cfg, input_shape)
        result["dcnv4_options"] = DCNv4Options.from_cfg(cfg)
        return result

    def layers(self, features, mask=None, targets=None):
        mask_features, _, multi_scale_features = self.pixel_decoder.forward_features(features, mask)
        if len(multi_scale_features) != len(self.dcnv4_feature_strides):
            raise RuntimeError("Native pixel decoder feature count changed")
        original = multi_scale_features[self.dcnv4_level_index]
        if self._source_feature is not None and original.shape[-2:] != features[self._source_feature].shape[-2:]:
            raise RuntimeError("Selected feature does not have the expected backbone stride/shape")
        enhanced = list(multi_scale_features)  # do not mutate the original list
        enhanced[self.dcnv4_level_index] = self.dcnv4_adapter(original)
        return self.predictor(enhanced, mask_features, mask, targets=targets)
