"""Run CPU acceptance and preserve the exact tested source fingerprints."""
import argparse
from pathlib import Path
import unittest
from polypbench.io import sha256, write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True)
args = parser.parse_args()
root = Path.cwd()
suite = unittest.TestSuite()
for pattern in ("test_candidate_stage*.py", "test_initialization_intervention.py", "test_quality_diagnostic.py"):
    suite.addTests(unittest.defaultTestLoader.discover(str(root/"tests"), pattern=pattern))
result = unittest.TextTestRunner(verbosity=2).run(suite)
paths = ["polypbench/analysis/initialization_intervention.py", "polypbench/analysis/initialization_queue.py",
    "polypbench/analysis/candidate_stage.py", "tests/test_candidate_stage.py", "tests/test_initialization_intervention.py"]
write_json(args.output, {"status": "passed" if result.wasSuccessful() else "failed",
    "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
    "training": False, "gpu_forward": False, "source_sha256": {p: sha256(root/p) for p in paths}})
raise SystemExit(0 if result.wasSuccessful() else 1)
