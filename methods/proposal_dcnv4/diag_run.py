"""Instrumented train entry for the DCNv4 combination (train only).

Same as methods.scda_proposal.diag_run but routes the model process to the
DCNv4 worker module. Predict stays on the standard methods.proposal_dcnv4.run.
"""
import os
import sys


def main():
    import methods.proposal_dcnv4.run as runmod

    original_command = runmod.command

    def diag_command(cfg, snapshot, output, mode, split="val", checkpoint=None):
        if mode != "train":
            return original_command(cfg, snapshot, output, mode, split, checkpoint)
        result = [cfg["runtime"]["python"], "-m", "methods.scda_proposal.diag_worker",
                  "--experiment", str(snapshot), "--output", str(output),
                  "--mode", mode, "--split", split,
                  "--worker-module", "methods.proposal_dcnv4.worker",
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
