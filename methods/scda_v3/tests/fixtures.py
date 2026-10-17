import torch
from methods.scda_proposal.structures import SCDATarget


def fixture(shapes=((16, 16), (8, 8), (4, 4), (2, 2)), canvas=(64, 64), gt_count=1):
    anchors, starts, total = [], [], 0
    for h, w in shapes:
        starts.append(total); total += h * w
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        anchors.append(torch.stack(((x + .5) / w, (y + .5) / h,
                                     torch.full_like(x, .2, dtype=torch.float32),
                                     torch.full_like(y, .2, dtype=torch.float32)), -1).reshape(-1, 4))
    anchors = torch.cat(anchors)[None]
    memory = torch.randn(1, total, 16, requires_grad=True)
    prior = torch.logit(anchors)
    payload = {"raw_cls": torch.zeros(1, total, 1, requires_grad=True),
               "raw_box_sigmoid": anchors.clone().requires_grad_(), "base_anchor": anchors,
               "topk_indices": torch.tensor([[0, 1]]), "spatial_shapes": torch.tensor(shapes),
               "level_start_index": torch.tensor(starts), "mask_flatten": torch.zeros(1, total, dtype=torch.bool),
               "valid_token_mask": torch.ones(1, total, dtype=torch.bool),
               "output_memory": memory, "proposal_unsigmoid": prior}
    mask = torch.zeros(gt_count, *canvas, dtype=torch.bool)
    mask[:, 8:-8, 8:-8] = True
    target = SCDATarget(torch.zeros(gt_count, dtype=torch.long),
                        torch.tensor([[.5, .5, .875, .875]]).repeat(gt_count, 1), mask,
                        torch.ones(gt_count, dtype=torch.bool), torch.arange(gt_count),
                        torch.ones(*canvas, dtype=torch.bool), canvas, original_gt_count=gt_count)
    return payload, target
