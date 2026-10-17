"""Per-step local assignment records, including fallback and no-candidate GTs."""
from dataclasses import asdict
import json
from pathlib import Path
import torch
from detectron2.utils import comm
from detectron2.utils.events import get_event_storage
from methods.scda_proposal.diagnostics import parameter_audit


def record_audit(assignments, counts, losses):
    storage = get_event_storage()
    images = []
    for assignment in assignments:
        images.append({"tokens": assignment.reg.token_indices.tolist(),
                       "gt_indices": assignment.reg.gt_indices.tolist(), "audit": assignment.audit})
    record = {"iteration": storage.iter, "global_valid_gt": float(counts[0]), "images": images,
              "loss_v3_box": float(losses["loss_v3_box"].detach())}
    storage.put_scalar("v3/valid_gt_global", record["global_valid_gt"], smoothing_hint=False)
    storage.put_scalar("v3/selected_tokens_local", sum(len(row["tokens"]) for row in images), smoothing_hint=False)
    directory = getattr(storage, "v3_output_dir", None)
    if directory:
        path = Path(directory) / f"v3_assignments_rank{comm.get_rank()}.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")


def optimizer_audit(model, optimizer):
    audit = parameter_audit(model, optimizer)
    aux = {id(p): name for name, p in model.named_parameters() if name.startswith("aux_box_head.")}
    audit["v3_auxiliary_groups"] = [
        {"names": [aux[id(p)] for p in group["params"] if id(p) in aux],
         "lr": group["lr"], "weight_decay": group["weight_decay"]}
        for group in optimizer.param_groups if any(id(p) in aux for p in group["params"])]
    audit["v3_auxiliary_parameters"] = sum(p.numel() for p in model.aux_box_head.parameters()) if aux else 0
    return audit
