"""Reproducible CPU acceptance; exclusive output, no training or CUDA context."""
import argparse
import contextlib
import datetime as dt
import importlib
import os
from pathlib import Path
import sys
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() or output.with_suffix(".log").exists():
        raise FileExistsError("Preserve existing V3 acceptance artifacts")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from polypbench.io import write_json, sha256
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root/"third_party/MaskDINO"))
    modules = ("test_geometry_assignment", "test_head_loss", "test_config_runtime", "test_model_cpu")
    suite = unittest.TestSuite()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".log").open("x", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        for module in modules:
            suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(
                importlib.import_module("methods.scda_v3.tests."+module)))
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    from .tests.test_model_cpu import CPU_ALIGNMENT_EVIDENCE
    from .run import adapter_sources
    from .experiment import load_experiment
    report = {"timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), "python": sys.version,
              "torch": torch.__version__, "tests_run": result.testsRun,
              "passed": result.wasSuccessful() and not torch.cuda.is_initialized(),
              "cuda_context_created": torch.cuda.is_initialized(),
              "failures": [{"test": str(case), "traceback": trace} for case, trace in result.failures],
              "errors": [{"test": str(case), "traceback": trace} for case, trace in result.errors],
              "skipped": [{"test": str(case), "reason": reason} for case, reason in result.skipped],
              "cpu_alignment": CPU_ALIGNMENT_EVIDENCE,
              "source_sha256": adapter_sources(load_experiment(root/"methods/scda_v3/configs/v3a.yaml")),
              "scope": "Canonical R50 construction/optimizer; reduced CPU forward/backward, DN off; CPU autocast; two-process Gloo",
              "not_run": "CUDA FP16/AMP, full native DN train forward, real-image GPU precheck, optimizer steps and formal training"}
    write_json(output, report)
    print({key: report[key] for key in ("passed", "tests_run", "cuda_context_created")})
    if not report["passed"]: print(report["failures"], report["errors"])
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__": main()
