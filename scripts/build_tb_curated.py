"""Rebuild the curated TensorBoard view (runs/_tb_all) with human-readable
run labels (incl. final AP when available).

Copies each run's events file into runs/_tb_all/<label>/ (real copies, not
symlinks: TensorBoard does not follow symlinked logdir subdirectories).
Re-runnable; the view is rebuilt from scratch each time.
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
TB = RUNS / "_tb_all"

LABELS = {
    "maskdino-official-pad0-a6000-b16-s42": "Clean-A_4e-4_s42",
    "scda-v2-pad0-a6000-b16-s42": "F-v2_cls+box_4e-4_s42",
    "scda-freg-pad0-a6000-b16-s42": "F-Reg_4e-4_s42",
    "scda-fcls-pad0-a6000-b16-s42": "F-Cls_4e-4_s42",
    "scda-fcls-safe-pad0-a6000-b16-s42": "F-Cls-Safe_4e-4_s42",
    "scda-freg-quality-pad0-a6000-b16-s42": "F-Reg-Quality_4e-4_s42",
    "scda-freg-quality-pad0-a6000-b16-s43": "F-Reg-Quality_4e-4_s43",
    "scda-qboxonly-pad0-a6000-b16-s42": "Q-BoxOnly_4e-4_s42",
    "proposal-conv-fv2-pad0-a6000-s42": "old-batch4_Conv+F-v2_s42",
    "proposal-dcn-fv2-pad0-a6000-s42": "old-batch4_DCN+F-v2_s42",
    "proposal-conv-fv2-kvasir-a6000-s42": "old-Conv+F-v2_kvasir_s42",
    "proposal-dcn-fv2-kvasir-a6000-s42": "old-DCN+F-v2_kvasir_s42",
    "scda-freg-quality-pad0-a6000-b16-s42-diag": "DIAG_F-Reg-Quality_4e-4_NO-CLIP",
    "scda-boxgate-iou-pad0-a6000-b16-s42-diag": "DIAG_Box-Gated_4e-4_NO-CLIP",
    "scda-freg-quality-pad0-a6000-b16-s42-lr1e4": "F-Reg-Quality_1e-4_s42",
    "proposal-dcn-freg-quality-pad0-a6000-b16-s42-lr1e4": "DCN-F-Reg-Quality_1e-4_s42",
    "maskdino-official-pad0-a6000-b16-s42-lr1e4": "Clean-A_1e-4_s42",
    "proposal-dcn-only-pad0-a6000-b16-s42-lr1e4": "DCNv4-only_1e-4_s42",
}


def final_ap(run_dir):
    mp = run_dir / "predict_val" / "metrics.json"
    if not mp.exists():
        return None
    try:
        return json.loads(mp.read_text())["metrics"]["AP"] * 100
    except Exception:
        return None


def main():
    if TB.exists():
        shutil.rmtree(TB)
    TB.mkdir(parents=True)
    built = []
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        if d.name.startswith("probe-"):  # 50-iter health probes are not formal runs
            continue
        events = sorted((d / "train").glob("events.out.tfevents.*"))
        if not events:
            continue
        label = LABELS.get(d.name, d.name)
        ap = final_ap(d)
        if ap is not None:
            label += f"_AP{ap:.1f}"
        dest = TB / label
        dest.mkdir()
        for ev in events:
            shutil.copy(ev, dest / ev.name)
        built.append(label)
    print(f"{TB} : {len(built)} runs")
    for label in built:
        n_events = len(list((TB / label).glob("events.*")))
        print(f"  {label}  (events files: {n_events})")


if __name__ == "__main__":
    main()
