"""Native LSJ geometry with zero padding for binary instance masks.

Detectron2 defaults FixedSizeCrop.seg_pad_value to the semantic ignore label
255. RLE instance masks and the ones-map used to track valid pixels cannot
use that value: both later become boolean tensors, making 255 foreground.
Keep every native image transform and its image pad value unchanged.
"""
from detectron2.data import transforms as T
from maskdino.data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import build_transform_gen


def build_instance_lsj_augmentations(cfg):
    augmentations = build_transform_gen(cfg, True)
    for augmentation in augmentations:
        if isinstance(augmentation, T.FixedSizeCrop):
            augmentation.seg_pad_value = 0
    return augmentations
