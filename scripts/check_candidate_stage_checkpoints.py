"""CPU construction and strict checkpoint loading; no model forward or CUDA."""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "third_party/MaskDINO"))
    import torch
    import train_net  # noqa: F401
    from polypbench.io import read_json, write_json, sha256
    from polypbench.analysis.candidate_stage import remap_experiment
    from methods.mssl.config import build_cfg
    from detectron2.modeling import build_model
    torch.set_num_threads(1)
    queue_root = Path(args.queue_root)
    rows = []
    for label in ("baseline", "mssl"):
        job = read_json(queue_root / f"{label}.job.json")
        for key in ("experiment", "checkpoint"):
            if sha256(job[key]) != job["expected_sha256"][key]:
                raise RuntimeError(f"Input drift: {label}/{key}")
        experiment = remap_experiment(read_json(job["experiment"]), root)
        cfg = build_cfg(experiment, queue_root, device="cpu")
        model = build_model(cfg)
        state = torch.load(job["checkpoint"], map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        decoder = model.sem_seg_head.predictor
        rows.append({"model": label, "strict_load": True, "device": "cpu", "model_forward": False,
            "checkpoint_sha256": job["expected_sha256"]["checkpoint"],
            "decoder_layers": decoder.num_layers, "candidate_feature_levels": decoder.num_feature_levels,
            "initialize_box_type": decoder.initialize_box_type})
        del state, model
    write_json(queue_root / "checkpoint_cpu_loads.json", rows)
    print(rows)


if __name__ == "__main__":
    main()
