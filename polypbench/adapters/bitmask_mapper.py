"""RLE-compatible LSJ data plumbing, preserving native MaskDINO augmentation.

Uses native image resize/crop/padding with zero instance-mask padding.
training_boxes="annotations" preserves
supplied box supervision; legacy "mask" reproduces mask-derived upstream boxes.
BitMasks avoids polygon conversion loss, including holes in Kvasir masks.
"""
import copy

import numpy as np
import torch
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks
from .instance_lsj import build_instance_lsj_augmentations


class BitmaskLSJMapper:
    def __init__(self, cfg, training_boxes="mask"):
        self.augmentations = build_instance_lsj_augmentations(cfg)
        self.image_format = cfg.INPUT.FORMAT
        self.training_boxes = training_boxes

    def __call__(self, original):
        record = copy.deepcopy(original)
        image = utils.read_image(record["file_name"], format=self.image_format)
        utils.check_image_size(record, image)
        padding = np.ones(image.shape[:2], dtype=np.uint8)
        image, transforms = T.apply_transform_gens(self.augmentations, image)
        padding = ~transforms.apply_segmentation(padding).astype(bool)
        shape = image.shape[:2]
        record["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        record["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding))
        if "local_semantic_rle" in record:
            from pycocotools import mask as mask_util
            semantic = mask_util.decode(record.pop("local_semantic_rle"))
            semantic = transforms.apply_segmentation(semantic)
            semantic[padding] = 0
            record["local_semantic_mask"] = torch.as_tensor(np.ascontiguousarray(semantic > 0))
        annotations = []
        for obj in record.pop("annotations", []):
            obj.pop("keypoints", None)
            if not obj.get("iscrowd", 0):
                annotations.append(utils.transform_instance_annotations(obj, transforms, shape))
        instances = utils.annotations_to_instances(annotations, shape, mask_format="bitmask")
        if not instances.has("gt_masks"):
            instances.gt_masks = BitMasks(torch.zeros((0, *shape), dtype=torch.bool))
        if self.training_boxes == "mask":
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        # Supplied box supervision is transformed geometrically, never recomputed
        # from masks. A crop can erase mask foreground while leaving a valid box.
        instances = utils.filter_empty_instances(instances, by_mask=self.training_boxes != "annotations")
        instances.gt_masks = instances.gt_masks.tensor
        record["instances"] = instances
        return record
