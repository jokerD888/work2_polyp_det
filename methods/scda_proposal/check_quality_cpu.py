"""Reproducible CPU contracts for quality+DCNv4 and calibrated Box-Gated IoU."""
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
    if output.exists():
        raise FileExistsError(f"Preserve previous CPU report: {output}")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from polypbench.io import write_json, sha256
    root = Path(__file__).resolve().parents[2]
    modules = ("methods.scda_proposal.tests.test_cpu",
               "methods.scda_proposal.tests.test_box_gate",
               "methods.scda_proposal.tests.test_box_calibration_config",
               "methods.scda_proposal.tests.test_precheck_policy",
               "methods.proposal_dcnv4.tests.test_cpu",
               "methods.proposal_dcnv4.tests.test_conv_control",
               "methods.proposal_dcnv4.tests.test_quality_config",
               "methods.proposal_dcnv4.tests.test_quality_preparation")
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.with_suffix(".log")
    with log.open("x", encoding="utf-8") as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        loader = unittest.defaultTestLoader
        suite = unittest.TestSuite(loader.loadTestsFromModule(importlib.import_module(name)) for name in modules)
        suite.addTests(loader.discover(str(root / "tests")))
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    created = torch.cuda.is_initialized()
    sources = [path for directory in (root / "methods/scda_proposal", root / "methods/proposal_dcnv4")
               for path in directory.rglob("*") if path.is_file() and path.suffix in (".py", ".yaml")]
    sources += [root / "configs/maskdino_kvasir_official_b16.yaml", root / "scripts/calibrate_scda_box_gate.py"]
    report = {"timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
              "python": sys.version, "torch": torch.__version__, "tests_run": result.testsRun,
              "passed": result.wasSuccessful() and not created, "cuda_context_created": created,
              "failures": [{"test": str(case), "traceback": trace} for case, trace in result.failures],
              "errors": [{"test": str(case), "traceback": trace} for case, trace in result.errors],
              "skipped": [{"test": str(case), "reason": reason} for case, reason in result.skipped],
              "source_sha256": {p.relative_to(root).as_posix(): sha256(p) for p in sorted(sources)},
              "log": str(log),
              "scope": "Real CPU model construction/optimizer/Conv; DCNv4 uses TokenProbe test double only",
              "gpu_or_training": "NOT RUN; real-image rho and official DCNv4 CUDA validation remain on training machine"}
    write_json(output, report)
    print({key: report[key] for key in ("passed", "tests_run", "cuda_context_created")})
    if not report["passed"]:
        print({"failures": report["failures"], "errors": report["errors"]})
    print(output)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
