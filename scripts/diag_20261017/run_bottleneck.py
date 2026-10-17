"""Experiment B1 (collector): val-split detection-bottleneck diagnostics.

Native eval forward per val image (no NMS/TTA/filtering) reproduces the
predict pipeline's 100-detection Instances. A parallel replica recomputes the
two-stage init path (encoder proposals -> top-300 -> interm -> mask2box) with
the model's OWN modules in eval mode and must reproduce the native final
outputs per image (gates below) — only then are the exposed internals used.

Identity gates recorded per run:
  replica_diff       max |native - replica| on final logits/boxes
  det_box_diff       max |native - replica| on top-100 detection boxes
  det_score_diff     max |native - replica| on detection scores
Final COCO AP identity (vs recorded predict_val metrics) is checked in the
analyzer. All GT usage is offline, after predictions.
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/home/ouc1/code/work2_polyp_det/work2_polyp_det")
from harness_diag import CLEAN_A_BEST, FREG_BEST, build_model, json_dump, load_weights, seed_all

OUT = HERE
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def replica_two_stage_init(predictor, mask_features, x):
    """Mirror of native maskdino_decoder.forward two-stage block (eval mode)."""
    from maskdino.utils.utils import gen_encoder_output_proposals, inverse_sigmoid
    from maskdino.utils import box_ops
    device = x[0].device
    masks = [torch.zeros((s.size(0), s.size(2), s.size(3)), device=device, dtype=torch.bool)
             for s in x]
    src_flatten, mask_flatten, spatial_shapes = [], [], []
    for i in range(predictor.num_feature_levels):
        idx = predictor.num_feature_levels - 1 - i
        src_flatten.append(predictor.input_proj[idx](x[idx]).flatten(2).transpose(1, 2))
        mask_flatten.append(masks[i].flatten(1))
        spatial_shapes.append(x[idx].shape[-2:])
    src_flatten = torch.cat(src_flatten, 1)
    mask_flatten = torch.cat(mask_flatten, 1)
    spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(1),
                                   spatial_shapes.prod(1).cumsum(0)[:-1]))
    output_memory, output_proposals = gen_encoder_output_proposals(
        src_flatten, mask_flatten, spatial_shapes)
    output_memory = predictor.enc_output_norm(predictor.enc_output(output_memory))
    enc_cls = predictor.class_embed(output_memory)
    enc_box_unsig = predictor._bbox_embed(output_memory) + output_proposals
    topk_proposals = torch.topk(enc_cls.max(-1)[0], predictor.num_queries, dim=1)[1]
    gather4 = topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
    gatherh = topk_proposals.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1])
    ref_undetach = torch.gather(enc_box_unsig, 1, gather4)
    tgt_undetach = torch.gather(output_memory, 1, gatherh)
    _, outputs_mask = predictor.forward_prediction_heads(
        tgt_undetach.transpose(0, 1), mask_features)
    if predictor.initialize_box_type == "mask2box":
        flat = outputs_mask.detach().flatten(0, 1)
        h, w = outputs_mask.shape[-2:]
        init = box_ops.masks_to_boxes(flat > 0)
        init = box_ops.box_xyxy_to_cxcywh(init) / torch.as_tensor(
            [w, h, w, h], dtype=torch.float, device=device)
        init = inverse_sigmoid(init.reshape(outputs_mask.shape[0], outputs_mask.shape[1], 4))
    else:
        init = ref_undetach.detach()
    return {
        "enc_cls": enc_cls, "enc_box": enc_box_unsig.sigmoid(),
        "topk": topk_proposals, "raw_top300": ref_undetach.sigmoid(),
        "init_unsigmoid": init, "init_boxes": init.sigmoid(),
        "src": src_flatten, "mask_flatten": mask_flatten,
        "spatial_shapes": spatial_shapes, "level_start_index": level_start_index,
        "tgt": tgt_undetach.detach(), "mask_features": mask_features,
        "interm_masks": outputs_mask,
    }


def replica_decoder(predictor, r):
    hs, references = predictor.decoder(
        tgt=r["tgt"].transpose(0, 1), memory=r["src"].transpose(0, 1),
        memory_key_padding_mask=r["mask_flatten"], pos=None,
        refpoints_unsigmoid=r["init_unsigmoid"].detach().transpose(0, 1),
        level_start_index=r["level_start_index"], spatial_shapes=r["spatial_shapes"],
        valid_ratios=torch.ones(r["src"].shape[0], len(r["spatial_shapes"]), 2,
                                device=r["src"].device), tgt_mask=None)
    out_class, out_mask = predictor.forward_prediction_heads(
        hs[-1].transpose(0, 1), r["mask_features"], True)
    out_boxes = predictor.pred_box(references, hs, r["init_unsigmoid"].sigmoid())
    return {"pred_logits": out_class, "pred_masks": out_mask, "pred_boxes": out_boxes[-1]}


def instance_replica(meta, out_rep, images, record):
    """Verbatim mirror of MaskDINO.forward's eval postprocessing (instance_on
    with sem_seg_postprocess_before_inference=True) applied to the replica
    outputs, plus the (query, cls, mask_q) decomposition of each detection."""
    import torch.nn.functional as F
    image_size = images.image_sizes[0]
    height, width = record["height"], record["width"]
    mask_pred_results = F.interpolate(out_rep["pred_masks"],
                                      size=images.tensor.shape[-2:],
                                      mode="bilinear", align_corners=False)
    mask_cls_result = out_rep["pred_logits"][0]
    mask_box_result = out_rep["pred_boxes"][0]
    new_size = mask_pred_results.shape[-2:]
    from detectron2.modeling.postprocessing import sem_seg_postprocess
    mask_pred_result = sem_seg_postprocess(mask_pred_results[0], image_size, height, width)
    mask_cls_result = mask_cls_result.to(mask_pred_result)
    height_n = new_size[0] / image_size[0] * height
    width_n = new_size[1] / image_size[1] * width
    mask_box_result = mask_box_result.to(mask_pred_result)
    mask_box_result = meta.box_postprocess(mask_box_result, height_n, width_n)
    inst = meta.instance_inference(mask_cls_result, mask_pred_result, mask_box_result)
    order = torch.argsort(inst.scores.cpu(), descending=True)
    num_classes = meta.sem_seg_head.num_classes
    flat = out_rep["pred_logits"][0].sigmoid().flatten()
    tk_scores, tk_idx = flat.topk(len(inst))
    cls_factor = tk_scores.cpu()
    mask_q = (inst.scores.cpu() / cls_factor.clamp(min=1e-12)).clamp(max=1.0)
    return {"boxes_xyxy": inst.pred_boxes.tensor[order].cpu(),
            "scores": inst.scores[order].cpu(),
            "labels": inst.pred_classes[order].cpu(),
            "cls_factor": cls_factor[order], "mask_quality": mask_q[order],
            "query": (tk_idx // num_classes)[order].cpu()}


def iou_matrix(preds, gts):
    """(G, N) IoU, preds N×4 and gts G×4 in the same pixel coords."""
    preds, gts = preds.detach().cpu().float(), gts.detach().cpu().float()
    if preds.numel() == 0 or gts.numel() == 0:
        return torch.zeros((gts.shape[0], preds.shape[0]))
    lt = torch.maximum(gts[:, None, :2], preds[None, :, :2])
    rb = torch.minimum(gts[:, None, 2:], preds[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    ap = (preds[:, 2] - preds[:, 0]).clamp(min=0) * (preds[:, 3] - preds[:, 1]).clamp(min=0)
    ag = (gts[:, 2] - gts[:, 0]).clamp(min=0) * (gts[:, 3] - gts[:, 1]).clamp(min=0)
    return inter / (ag[:, None] + ap[None, :] - inter).clamp(min=1e-9)


def norm_to_orig_xyxy(box_norm, width, height, aug_hw, pad_hw):
    """Native convention (maskdino.py lines 325-327 + box_postprocess):
    original px = norm * (H_p/H_a * height, W_p/W_a * width)."""
    h_n = pad_hw[0] / aug_hw[0] * height
    w_n = pad_hw[1] / aug_hw[1] * width
    xy = torch.stack([box_norm[:, 0] - box_norm[:, 2] / 2,
                      box_norm[:, 1] - box_norm[:, 3] / 2,
                      box_norm[:, 0] + box_norm[:, 2] / 2,
                      box_norm[:, 1] + box_norm[:, 3] / 2], -1)
    return xy * torch.tensor([w_n, h_n, w_n, h_n], device=box_norm.device)


def main():
    seed_all(42)
    model, cfg, experiment = build_model()
    from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
    from detectron2.data.datasets import load_coco_json
    entry = experiment["dataset"]["val"]
    name = "diag_val"
    if name not in DatasetCatalog.list():
        DatasetCatalog.register(name, lambda: load_coco_json(
            entry["annotations"], entry["image_root"], name))
        MetadataCatalog.get(name).set(evaluator_type="coco")
    gt_dicts = load_coco_json(entry["annotations"], entry["image_root"], name)
    gt_by_id = {}
    for g in gt_dicts:
        rows = []
        for a in g.get("annotations", []):
            if a.get("iscrowd", 0):
                continue
            x, y, w, h = a["bbox"]
            rows.append({"xyxy": [x, y, x + w, y + h],
                         "area": float(a.get("area", w * h))})
        gt_by_id[g["image_id"]] = rows
    loader = build_detection_test_loader(cfg, name)

    records = list(loader)
    if LIMIT:
        records = records[:LIMIT]

    for label, ckpt in [("CleanA", CLEAN_A_BEST), ("FReg", FREG_BEST)]:
        load_weights(model, ckpt)
        model.eval()
        per_image, detections = [], []
        max_replica_diff = max_det_box_diff = max_det_score_diff = 0.0
        with torch.no_grad():
            for record_batch in records:
                record = record_batch[0]
                inst = model([record])[0]["instances"]
                order = torch.argsort(inst.scores.cpu(), descending=True)
                det_boxes = inst.pred_boxes.tensor[order].cpu()
                det_scores = inst.scores[order].cpu()
                det_labels = inst.pred_classes[order].cpu()
                for k in range(len(det_scores)):
                    x0, y0, x1, y1 = det_boxes[k].tolist()
                    detections.append({"image_id": record["image_id"], "category_id": 1,
                                       "bbox": [x0, y0, x1 - x0, y1 - y0],
                                       "score": round(float(det_scores[k]), 6)})
                from detectron2.structures import ImageList
                img = (record["image"].to(model.device) - model.pixel_mean) / model.pixel_std
                images = ImageList.from_tensors([img], model.size_divisibility)
                features = model.backbone(images.tensor)
                out_native, _ = model.sem_seg_head(features)
                mask_features, _, multi = \
                    model.sem_seg_head.pixel_decoder.forward_features(features, None)
                r = replica_two_stage_init(model.sem_seg_head.predictor, mask_features, multi)
                out_rep = replica_decoder(model.sem_seg_head.predictor, r)
                replica_diff = max(
                    (out_native["pred_logits"] - out_rep["pred_logits"]).abs().max().item(),
                    (out_native["pred_boxes"] - out_rep["pred_boxes"]).abs().max().item())
                max_replica_diff = max(max_replica_diff, replica_diff)
                rep = instance_replica(model, out_rep, images, record)
                if len(det_scores) == len(rep["scores"]):
                    det_box_diff = (det_boxes - rep["boxes_xyxy"]).abs().max().item()
                    det_score_diff = (det_scores - rep["scores"]).abs().max().item()
                else:
                    det_box_diff = det_score_diff = float("nan")
                max_det_box_diff = max(max_det_box_diff, det_box_diff)
                max_det_score_diff = max(max_det_score_diff, det_score_diff)

                gt_rows = gt_by_id[record["image_id"]]
                gt_boxes = torch.tensor([g["xyxy"] for g in gt_rows]).reshape(-1, 4)
                width, height = record["width"], record["height"]
                aug_hw = images.image_sizes[0]
                pad_hw = images.tensor.shape[-2:]
                tok_orig = norm_to_orig_xyxy(r["enc_box"][0].float(), width, height, aug_hw, pad_hw)
                e_orig = norm_to_orig_xyxy(r["raw_top300"][0].float(), width, height, aug_hw, pad_hw)
                m_orig = norm_to_orig_xyxy(r["init_boxes"][0].float(), width, height, aug_hw, pad_hw)
                d_orig = norm_to_orig_xyxy(out_native["pred_boxes"][0].float(), width, height, aug_hw, pad_hw)
                iou_tok = iou_matrix(tok_orig, gt_boxes)
                iou_e = iou_matrix(e_orig, gt_boxes)
                iou_m = iou_matrix(m_orig, gt_boxes)
                iou_d = iou_matrix(d_orig, gt_boxes)
                cls_tok = torch.sigmoid(r["enc_cls"][0].float().max(-1)[0]).cpu()
                starts = r["level_start_index"].tolist()
                shapes = [tuple(s) for s in r["spatial_shapes"].tolist()]

                def token_level(t):
                    k = 0
                    for li, st in enumerate(starts):
                        if t >= st:
                            k = li
                    return f"{shapes[k][0]}x{shapes[k][1]}"

                stage1 = []
                topk0 = r["topk"][0].cpu()
                for g in range(len(gt_rows)):
                    iou_g = iou_tok[g]
                    order_t = torch.argsort(cls_tok, descending=True)
                    prefix_best = iou_g[order_t].cummax(0)[0]
                    best_t = int(iou_g.argmax())
                    inside = iou_g[topk0]
                    outside = iou_g.clone()
                    outside[topk0] = -1.0
                    row = {"gt": g, "oracle_best": round(float(iou_g.max()), 4),
                           "oracle_best_rank": int((cls_tok > cls_tok[best_t]).sum()),
                           "oracle_best_level": token_level(best_t),
                           "oracle_best_in_top300": bool((topk0 == best_t).any())}
                    for K in (1, 10, 50, 100, 300):
                        row[f"score_top{K}_best"] = round(float(prefix_best[min(K, len(prefix_best)) - 1]), 4)
                    for thr in (0.5, 0.75):
                        row[f"recall{int(thr*100)}_scoretop300"] = bool(
                            prefix_best[min(300, len(prefix_best) - 1)] >= thr)
                    row["best_inside_top300"] = round(float(inside.max()), 4)
                    row["best_outside_top300"] = round(float(outside.max()), 4)
                    stage1.append(row)

                per_image.append({
                    "image_id": record["image_id"],
                    "replica_diff": replica_diff,
                    "det_box_diff": det_box_diff, "det_score_diff": det_score_diff,
                    "gt_boxes": [g["xyxy"] for g in gt_rows],
                    "gt_area": [g["area"] for g in gt_rows],
                    "iouE300": [[round(v, 4) for v in iou_e[g].tolist()] for g in range(len(gt_rows))],
                    "iouM": [[round(v, 4) for v in iou_m[g].tolist()] for g in range(len(gt_rows))],
                    "iouD": [[round(v, 4) for v in iou_d[g].tolist()] for g in range(len(gt_rows))],
                    "cls_per_query": [round(v, 5) for v in cls_tok.tolist()],
                    "stage1": stage1,
                    "dets": {"boxes": [[round(v, 2) for v in b] for b in det_boxes.tolist()],
                             "scores": [round(v, 6) for v in det_scores.tolist()],
                             "labels": det_labels.tolist(),
                             "cls_factor": [round(v, 6) for v in rep["cls_factor"].tolist()],
                             "mask_q": [round(v, 6) for v in rep["mask_quality"].tolist()],
                             "query": rep["query"].tolist()},
                })
        gates = {"max_replica_diff": max_replica_diff,
                 "max_det_box_diff": max_det_box_diff,
                 "max_det_score_diff": max_det_score_diff,
                 "n_images": len(per_image)}
        json_dump(OUT / f"b1_raw_{label}.json", {"gates": gates, "images": per_image,
                                                 "detections": detections})
        print(f"{label}: {gates}", flush=True)
    print("B1_RAW_DONE")


if __name__ == "__main__":
    main()
