"""Shared harness for H1-A (stage box quality) and H2-A (gradient competition).

Builds the F-Reg-Quality LR=1e-4 experiment as an SCDAProposalMaskDINO model
with the payload-instrumented SCDAProposalDecoder (ENABLED=True so the payload
path executes, DIAGNOSTICS=True so decoder_init_boxes is exposed). No SCDA
loss is applied unless the calling script asks for it; loading either the
Clean A or the F-Reg-Quality checkpoint is parameter-compatible because both
instrumented classes add zero parameters/buffers.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path("/home/ouc1/code/work2_polyp_det/work2_polyp_det")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/MaskDINO"))

FREG_EXPERIMENT = ROOT / "runs/_launcher/lr1e4-a6000-s42/reference_quality_lr1e4_gpu0.yaml"
CLEAN_A_BEST = ROOT / "runs/maskdino-official-pad0-a6000-b16-s42-lr1e4/train/model_best.pth"
FREG_BEST = ROOT / "runs/scda-freg-quality-pad0-a6000-b16-s42-lr1e4/train/model_best.pth"
FREG_5224 = ROOT / "runs/scda-freg-quality-pad0-a6000-b16-s42-lr1e4/train/model_0005224.pth"
FREG_FINAL = ROOT / "runs/scda-freg-quality-pad0-a6000-b16-s42-lr1e4/train/model_0005499.pth"

# Decoder box loss keys of the native criterion (final + 9 aux layers), DN excluded.
DECODER_BOX_KEYS = ["loss_bbox"] + [f"loss_bbox_{i}" for i in range(9)]
DN_BOX_KEYS = ["loss_bbox_dn"] + [f"loss_bbox_dn_{i}" for i in range(9)]

# Parameter-name buckets for the modules named in the H2-A protocol.
# bbox_embed FIRST: the shared head is registered under
# predictor.decoder.bbox_embed.* (alias at decoder.py:161, and the decoder
# module precedes _bbox_embed in registration order, so named_parameters'
# dedup keeps the decoder-side alias). It must be matched before the generic
# transformer_decoder prefix or it is silently swallowed there.
MODULE_PREFIXES = [
    ("bbox_embed", "sem_seg_head.predictor._bbox_embed."),
    ("bbox_embed", "sem_seg_head.predictor.decoder.bbox_embed."),
    ("class_embed", "sem_seg_head.predictor.class_embed."),
    ("enc_output", "sem_seg_head.predictor.enc_output"),
    ("input_proj", "sem_seg_head.predictor.input_proj"),
    ("transformer_decoder", "sem_seg_head.predictor.decoder."),
    ("mask_embed", "sem_seg_head.predictor.mask_embed."),
    ("query_feat", "sem_seg_head.predictor.query_feat"),
    ("label_enc", "sem_seg_head.predictor.label_enc"),
    ("pixel_decoder", "sem_seg_head.pixel_decoder."),
    ("backbone", "backbone."),
]


def seed_all(seed=42):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_experiment():
    from methods.scda_proposal.experiment import load_experiment
    experiment = load_experiment(FREG_EXPERIMENT)
    return experiment


def build_model(device="cuda"):
    """SCDA-instrumented model carrying F-Reg-Quality's exact protocol config."""
    import torch
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from detectron2.projects.deeplab import add_deeplab_config
    import train_net as native
    from methods.scda_proposal.config import add_scda_config
    from methods.scda_proposal.integration import register

    experiment = build_experiment()
    register(experiment["repo"]["path"])
    cfg = get_cfg()
    add_deeplab_config(cfg)
    native.add_maskdino_config(cfg)
    add_scda_config(cfg)
    cfg.merge_from_file(experiment["model"]["config"])
    for key, value in experiment["overrides"].items():
        cfg.merge_from_list([key, repr(value)])
    cfg.MODEL.META_ARCHITECTURE = "SCDAProposalMaskDINO"
    cfg.MODEL.MaskDINO.TRANSFORMER_DECODER_NAME = "SCDAProposalDecoder"
    cfg.MODEL.SCDA.ENABLED = True          # payload path in the decoder
    cfg.MODEL.SCDA.DIAGNOSTICS = True      # expose decoder_init_boxes
    cfg.MODEL.DEVICE = device
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = experiment["model"]["num_classes"]
    cfg.SEED = experiment["seed"]
    cfg.TEST.AUG.ENABLED = False
    cfg.freeze()
    model = build_model(cfg)
    return model, cfg, experiment


def load_weights(model, path):
    from detectron2.checkpoint import DetectionCheckpointer
    import os
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    if path is None:
        return {"loaded": "none(random init)"}
    result = DetectionCheckpointer(model).load(str(path))
    return {"loaded": str(path)}


def register_dataset(experiment, name="diag_train"):
    from detectron2.data import DatasetCatalog
    from detectron2.data.datasets import load_coco_json
    entry = experiment["dataset"]["train"]
    if name not in DatasetCatalog.list():
        DatasetCatalog.register(name, lambda: load_coco_json(
            entry["annotations"], entry["image_root"], name,
            extra_annotation_keys=["id", "scda_reliable"]))
    return DatasetCatalog.get(name)


def map_fixed_records(cfg, experiment, dataset_dicts, seed=42):
    """One deterministic LSJ-augmented record per train image, in file order."""
    from methods.scda_proposal.mapper import SCDALSJMapper
    from methods.scda_proposal.config import SCDAOptions
    seed_all(seed)
    mapper = SCDALSJMapper(cfg, SCDAOptions().mask_policy)
    return [mapper(d) for d in dataset_dicts]


def make_batches(records, batch_size, count, seed=42, shuffle=True):
    """Deterministically shuffled fixed batches over the mapped records."""
    import random as _random
    rng = _random.Random(seed)
    order = list(range(len(records)))
    if shuffle:
        rng.shuffle(order)
    return [order[i * batch_size:(i + 1) * batch_size] for i in range(count)]


def to_model_inputs(model, batch_records):
    """Replicates MaskDINO.forward's image preprocessing; returns (images, gt_instances)."""
    import torch
    from detectron2.structures import ImageList
    images = [r["image"].to(model.device) for r in batch_records]
    images = [(x - model.pixel_mean) / model.pixel_std for x in images]
    images = ImageList.from_tensors(images, model.size_divisibility)
    gt_instances = [r["instances"].to(model.device) for r in batch_records]
    return images, gt_instances


def scda_targets(model, batch_records, gt_instances, images, native_targets):
    return model._scda_targets(batch_records, gt_instances, native_targets, images)


def module_of(name):
    for bucket, prefix in MODULE_PREFIXES:
        if name.startswith(prefix):
            return bucket
    return "other"


def cxcywh_to_xyxy_px(boxes, w, h):
    """Normalized cxcywh -> pixel xyxy on the padded canvas."""
    import torch
    cx, cy, bw, bh = boxes.unbind(-1)
    return torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1) * \
        torch.tensor([w, h, w, h], device=boxes.device, dtype=boxes.dtype)


def pairwise_iou_best(candidates_px, gt_px):
    """candidates N,4; gt G,4 pixel xyxy -> G,N IoU."""
    import torch
    if candidates_px.numel() == 0 or gt_px.numel() == 0:
        return torch.zeros((gt_px.shape[0], candidates_px.shape[0]), device=candidates_px.device)
    lt = torch.maximum(candidates_px[None, :, :2], gt_px[:, None, :2])
    rb = torch.minimum(candidates_px[None, :, 2:], gt_px[:, None, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_c = (candidates_px[:, 2] - candidates_px[:, 0]).clamp(min=0) * \
             (candidates_px[:, 3] - candidates_px[:, 1]).clamp(min=0)
    area_g = (gt_px[:, 2] - gt_px[:, 0]).clamp(min=0) * (gt_px[:, 3] - gt_px[:, 1]).clamp(min=0)
    return inter / (area_g[:, None] + area_c[None, :] - inter).clamp(min=1e-9)


def json_dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=float))
