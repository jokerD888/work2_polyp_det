"""Read numerical acceptance from interleaved training/evaluation JSON records."""
import json
import math
from pathlib import Path
from polypbench.io import read_json


def smoke_gate(run_root):
    root=Path(run_root)
    for stage in ("train","predict_val"):
        if read_json(root/stage/"state.json").get("status")!="completed":
            raise RuntimeError(f"Smoke {stage} did not complete")
    rows=[json.loads(line) for line in (root/"train/metrics.json").read_text().splitlines() if line.strip()]
    training=[row for row in rows if all(key in row for key in
        ("total_loss","loss_mssl","optimizer/successful_updates"))]
    if not training:
        raise RuntimeError("Smoke has no training metrics")
    last=max(training,key=lambda row:row["iteration"])
    if last["iteration"]<15 or last["optimizer/successful_updates"]<1:
        raise RuntimeError("Smoke has insufficient iterations or no successful AMP update")
    if not all(math.isfinite(last[k]) for k in ("total_loss","loss_mssl")):
        raise RuntimeError("Smoke has nonfinite loss")
    read_json(root/"predict_val/metrics.json")
    return {"status":"passed","last_iteration":last["iteration"],
            "successful_updates":last["optimizer/successful_updates"],"total_loss":last["total_loss"],
            "loss_mssl":last["loss_mssl"],"checkpoint_reload_and_full_val_evaluation":True,
            "evaluation_path":str(root/"predict_val/metrics.json"),
            "scope":"numerical/runtime acceptance only, not method effectiveness"}
