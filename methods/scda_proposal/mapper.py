"""SCDA metadata on the same zero-mask-padding LSJ pipeline as the baseline."""
import copy

import numpy as np
import torch
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BitMasks
from polypbench.adapters.instance_lsj import build_instance_lsj_augmentations


class SCDALSJMapper:
    def __init__(self, cfg, mask_policy):
        self.augmentations = build_instance_lsj_augmentations(cfg)
        self.image_format = cfg.INPUT.FORMAT
        self.mask_policy = mask_policy

    def __call__(self, original):
        record = copy.deepcopy(original)
        original_annotations = [obj for obj in record.pop("annotations", []) if not obj.get("iscrowd", 0)]
        # Reliability is determined BEFORE augmentation/filtering. A two-box
        # original image cannot become SCDA-eligible after a crop erases one GT.
        single_instance = len(original_annotations) == 1
        image = utils.read_image(record["file_name"], format=self.image_format)
        utils.check_image_size(record, image)
        padding = np.ones(image.shape[:2], dtype=np.uint8)
        image, transforms = T.apply_transform_gens(self.augmentations, image)
        padding = ~transforms.apply_segmentation(padding).astype(bool)
        shape = image.shape[:2]
        record["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        record["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding))
        annotations, reliable, ids = [], [], []
        for index, obj in enumerate(original_annotations):
            obj.pop("keypoints", None)
            if self.mask_policy == "single_instance":
                valid_mask = single_instance and "segmentation" in obj
            elif self.mask_policy == "explicit":
                valid_mask = obj.get("scda_reliable") is True and "segmentation" in obj
            else:
                raise ValueError("Unapproved mask reliability policy for this Kvasir implementation")
            reliable.append(valid_mask)
            ids.append(obj.get("id", index))
            annotations.append(utils.transform_instance_annotations(obj, transforms, shape))
        instances = utils.annotations_to_instances(annotations, shape, mask_format="bitmask")
        if not instances.has("gt_masks"):
            instances.gt_masks = BitMasks(torch.zeros((0, *shape), dtype=torch.bool))
        instances.gt_scda_reliable = torch.as_tensor(reliable, dtype=torch.bool)
        instances.gt_scda_ids = torch.as_tensor(ids, dtype=torch.long)
        # Preserve the same transformed supplied bbox and filtering as baseline.
        instances = utils.filter_empty_instances(instances, by_mask=False)
        instances.gt_masks = instances.gt_masks.tensor
        record["instances"] = instances
        record["scda_source"] = "kvasir_official_bbox_clipped_semantic_mask"
        record["scda_original_gt_count"] = len(original_annotations)
        return record
