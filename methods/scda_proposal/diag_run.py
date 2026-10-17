"""Standard run entry routed through the instrumented diag_worker (train only).

Usage: python -m methods.scda_proposal.diag_run --config <yaml> --mode train --run-id <id>
Predict stays on the standard entry (methods.scda_proposal.run).
"""
import os
from pathlib import Path
import sys


def main():
    import methods.scda_proposal.run as runmod

    original_command = runmod.command

    def diag_command(cfg, snapshot, output, mode, split="val", checkpoint=None):
        if mode != "train":
            return original_command(cfg, snapshot, output, mode, split, checkpoint)
        result = [cfg["runtime"]["python"], "-m", "methods.scda_proposal.diag_worker",
                  "--experiment", str(snapshot), "--output", str(output),
                  "--mode", mode, "--split", split,
                  "--diag-level", os.environ.get("DIAG_LEVEL", "full")]
        if os.environ.get("DIAG_FORCE_TRIGGER_ITER"):
            result += ["--force-trigger-iter", os.environ["DIAG_FORCE_TRIGGER_ITER"]]
        if checkpoint:
            result += ["--checkpoint", str(checkpoint)]
        return result

    runmod.command = diag_command
    runmod.main()


if __name__ == "__main__":
    main()
