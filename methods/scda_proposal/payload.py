"""Expose native predictions without re-running shared prediction heads."""
import torch


def make_payload(raw_cls, raw_box_unsigmoid, base_proposals, topk_indices,
                 spatial_shapes, level_start_index, mask_flatten, output_memory,
                 decoder_init_boxes, diagnostics=False, init_from_mask=False):
    # Legal native proposals were not masked with inf by
    # gen_encoder_output_proposals(). Sigmoid widths alone cannot reveal that.
    raw_boxes = raw_box_unsigmoid.sigmoid()
    valid = (~mask_flatten & torch.isfinite(base_proposals).all(-1) &
             torch.isfinite(raw_cls.detach()).all(-1) &
             torch.isfinite(raw_box_unsigmoid.detach()).all(-1) &
             torch.isfinite(raw_boxes.detach()).all(-1) &
             (raw_boxes.detach()[..., 2:] > 0).all(-1))
    payload = {
        "raw_cls": raw_cls,
        "raw_box_sigmoid": raw_boxes,
        "base_anchor": base_proposals.detach().sigmoid(),
        "topk_indices": topk_indices.detach(),
        "spatial_shapes": spatial_shapes.detach(),
        "level_start_index": level_start_index.detach(),
        "mask_flatten": mask_flatten.detach(),
        "valid_token_mask": valid.detach(),
    }
    if diagnostics:
        # This is the matching-query reference AFTER native mask2box and BEFORE
        # DN queries are concatenated. It is not the raw selected proposal box.
        payload["decoder_init_boxes"] = decoder_init_boxes.detach()
        payload["init_from_mask"] = init_from_mask
        payload["output_memory"] = output_memory
    return payload
