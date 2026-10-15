"""Validate completed F-v2 evidence before a fresh full-budget run."""
import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from methods.scda_proposal.audit_summary import summarize_decoupling, validate_v2_assignments
from methods.scda_proposal.experiment import load_experiment
from methods.scda_proposal.run import adapter_sources
from polypbench.io import read_json, write_json, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", default="runs/_launcher/scda-proposal-v2-kvasir-s42")
    parser.add_argument("--evidence", default="runs/scda-v2-precheck-20261015")
    parser.add_argument("--smoke", default="runs/scda-proposal-v2-kvasir-smoke50")
    args = parser.parse_args()
    control, evidence, smoke = map(Path, (args.control, args.evidence, args.smoke))
    if (control / "acceptance.json").exists():
        raise FileExistsError("Preserve existing acceptance record")
    if read_json(control / "precheck_status.json")["status"] != "completed":
        raise ValueError("Precheck supervisor has not completed")
    report = read_json(evidence / "report.json")
    if report["status"] != "passed" or not all(check["passed"] for check in report["checks"].values()):
        raise ValueError("Real-model precheck did not pass")
    for filename in ("real_geometry.json", "post_smoke_gradients.json"):
        if not read_json(evidence / filename)["passed"]:
            raise ValueError(f"Failed {filename}")
    candidate_summaries = {}
    for filename in ("SCDA_DECOUPLING_EFFECTIVENESS.json", "mature_baseline_diagnostic.json"):
        summary = read_json(evidence / filename)
        validate_v2_assignments(summary["per_image"])
        candidate_summaries[filename] = summary["nonempty_valid_reliable_gt"]
        if not summary["nonempty_valid_reliable_gt"]["reg_transition_selected_count"]:
            raise ValueError("No medium-band supervision in diagnostic")
    if read_json(smoke / "train" / "state.json")["status"] != "completed":
        raise ValueError("Smoke has not completed")
    import json
    metrics = [json.loads(line) for line in (smoke / "train" / "metrics.json").read_text().splitlines()]
    training_rows = [row for row in metrics if "total_loss" in row]
    if not training_rows or max(row["iteration"] for row in training_rows) != 49:
        raise ValueError("Expected completed 50-iteration smoke")
    if any(not math.isfinite(value) for row in metrics for key, value in row.items()
           if (key.startswith("loss_") or key == "total_loss")):
        raise ValueError("Non-finite smoke loss")
    audits = [json.loads(line) for line in (smoke / "train" / "scda_audit.rank0.jsonl").read_text().splitlines()]
    if len(audits) != 50 or {row["iteration"] for row in audits} != set(range(50)):
        raise ValueError("Incomplete 50-step assignment audit")
    images = [image for row in audits for image in row["assignment"]]
    policy = validate_v2_assignments(images)
    cfg = load_experiment(control / "formal.yaml")
    current_sources = adapter_sources(cfg)
    if current_sources != report["method_sources"] or current_sources != read_json(smoke / "adapter_sources.json"):
        raise ValueError("Method source changed since verification/smoke; revalidate")
    baseline = read_json(Path(cfg["output_root"]) / "maskdino-official-s42" / "experiment.json")
    for key in ("seed", "training_budget", "model", "dataset", "overrides", "evaluation", "checkpoint_selection"):
        if cfg[key] != baseline[key]:
            raise ValueError(f"Full protocol differs from baseline in {key}")
    if cfg["scda"]["reg_selection"] != "core_plus_transition" or cfg["scda"]["diagnostics"]:
        raise ValueError("Need explicit F-v2, no per-step gradient diagnostics in formal config")
    record = {"passed": True, "formal_config_sha256": sha256(control / "formal.yaml"),
              "formal_base_sha256": sha256(control / "formal_base.yaml"),
              "method_sources": current_sources, "smoke_policy": policy,
              "smoke_summary": summarize_decoupling(images)["nonempty_valid_reliable_gt"],
              "candidate_summaries": candidate_summaries,
              "post_smoke_gradients_sha256": sha256(evidence / "post_smoke_gradients.json"),
              "note": "User-approved F-v2 rule and full budget; engineering acceptance only, no AP benefit claimed"}
    write_json(control / "acceptance.json", record)
    print(json.dumps({key: record[key] for key in ("passed", "smoke_policy", "smoke_summary")}, indent=2))


if __name__ == "__main__":
    main()
