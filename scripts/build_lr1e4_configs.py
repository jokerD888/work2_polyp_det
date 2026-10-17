"""Build frozen LR-1e-4 comparison configs from the real F-Reg-Quality chain.

Only SOLVER.BASE_LR changes (4e-4 -> 1e-4); everything else is inherited
bit-for-bit from the approved reference chain. Fails if any other field moves.
"""
import copy
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SRC = ROOT / "runs/_launcher/quality-boxgate-a6000-s42"
DST = ROOT / "runs/_launcher/lr1e4-a6000-s42"


def load(p):
    return yaml.safe_load(Path(p).read_text())


def flat_base(visible_gpus):
    base = load(SRC / "reference_base.yaml")
    base["name"] = "maskdino_quality_reference_b16_lr1e4"
    base["purpose"] = "lr_comparison_1e4"
    base["runtime"]["num_gpus"] = 1
    base["runtime"]["visible_gpus"] = str(visible_gpus)
    before = copy.deepcopy(base)
    base["overrides"]["SOLVER.BASE_LR"] = 0.0001
    diff = {k for k in set(before) | set(base)
            if k not in ("overrides", "runtime") and before.get(k) != base.get(k)} | {
        f"overrides.{k}" for k in set(before["overrides"]) | set(base["overrides"])
        if before["overrides"].get(k) != base["overrides"].get(k)} | {
        f"runtime.{k}" for k in set(before["runtime"]) | set(base["runtime"])
        if before["runtime"].get(k) != base["runtime"].get(k)}
    allowed = {"name", "purpose", "overrides.SOLVER.BASE_LR"}
    if visible_gpus != 0:
        allowed.add("runtime.visible_gpus")
    unexpected = diff - allowed
    if unexpected:
        raise SystemExit(f"unexpected base diffs: {sorted(unexpected)}")
    return base


def main():
    if DST.exists():
        raise SystemExit(f"{DST} exists")
    DST.mkdir(parents=True)
    for gpu in (0, 1):
        base = flat_base(gpu)
        (DST / f"reference_base_lr1e4_gpu{gpu}.yaml").write_text(yaml.safe_dump(base, sort_keys=False))
        rq = load(SRC / "reference_quality.yaml")
        rq["base_experiment"] = f"reference_base_lr1e4_gpu{gpu}.yaml"
        rq["name"] = "scda_freg_quality_reference_b16_lr1e4"
        (DST / f"reference_quality_lr1e4_gpu{gpu}.yaml").write_text(yaml.safe_dump(rq, sort_keys=False))
    # GPU1 also carries the DCNv4 combination on the same 1e-4 chain.
    dcn = load(SRC / "dcn_quality.yaml")
    dcn["base_experiment"] = "reference_quality_lr1e4_gpu1.yaml"
    dcn["name"] = "proposal_dcnv4_freg_quality_r50_kvasir_official_b16_lr1e4"
    (DST / "dcn_quality_lr1e4_gpu1.yaml").write_text(yaml.safe_dump(dcn, sort_keys=False))
    # audit trail
    files = sorted(p.name for p in DST.glob("*.yaml"))
    print({"dir": str(DST), "files": files})


if __name__ == "__main__":
    main()
