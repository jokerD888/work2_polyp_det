"""Additional attached memory and detached original proposal logits, train only."""
from methods.scda_proposal.payload import make_payload as make_legacy_payload


def make_payload(raw_cls, raw_box_unsigmoid, base_proposals, topk_indices,
                 spatial_shapes, level_start_index, mask_flatten, output_memory,
                 decoder_init_boxes, diagnostics=False, init_from_mask=False):
    result = make_legacy_payload(raw_cls, raw_box_unsigmoid, base_proposals, topk_indices,
                                 spatial_shapes, level_start_index, mask_flatten, output_memory,
                                 decoder_init_boxes, diagnostics=diagnostics,
                                 init_from_mask=init_from_mask)
    result["output_memory"] = output_memory
    result["proposal_unsigmoid"] = base_proposals.detach()
    return result
