"""Run CPU contracts with an auditable JSON result; no CUDA probing."""
import argparse
import datetime as dt
import os
from pathlib import Path
import platform
import sys
import unittest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from polypbench.io import write_json
    from .tests import test_cpu
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_cpu)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    created = torch.cuda.is_initialized()
    report = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "tests_run": result.testsRun, "failures": result.failures, "errors": result.errors,
        "skipped": result.skipped, "passed": result.wasSuccessful() and not created,
        "cuda_context_created": created,
        "operator_scope": "Explicit TokenProbe test double only; official CUDA operator NOT validated",
        "real_components": ["MaskDINO head", "full native/SCDA model construction",
                            "configuration", "native optimizer", "SCDA F-v2 preset inheritance"],
        "gpu_tests": "NOT_RUN_USER_REQUEST", "official_extension_build": "NOT_RUN_USER_REQUEST",
    }
    # unittest failures contain TestCase instances, which are not JSON objects.
    report["failures"] = [{"test": str(case), "traceback": trace} for case, trace in result.failures]
    report["errors"] = [{"test": str(case), "traceback": trace} for case, trace in result.errors]
    write_json(Path(args.output), report)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
