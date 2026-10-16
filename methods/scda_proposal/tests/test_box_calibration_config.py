"""Calibration identity is required, bound to train data and copied into cfg."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from methods.scda_proposal.config import SCDAOptions
from methods.scda_proposal.experiment import load_experiment
from methods.scda_proposal.box_gate_calibration import (fit_count_calibration,
                    IMAGENET_R50_SHA256, protocol_signature, source_identity)
from polypbench.config import load_config, fingerprint
from polypbench.io import sha256


ROOT = Path(__file__).resolve().parents[3]


class CalibrationConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        baseline = yaml.safe_load((ROOT / "configs/maskdino_kvasir_official_b16.yaml").read_text())
        baseline["project_root"] = str(ROOT)
        self.train = self.directory / "train.json"
        self.train.write_text("{}")
        baseline["dataset"]["train"]["annotations"] = str(self.train)
        self.baseline = self.directory / "base.yaml"
        self.baseline.write_text(yaml.safe_dump(baseline))
        options = SCDAOptions(enabled=True, reg_selection="quality_iou", lambda_cls=0.)
        cfg = load_config(self.baseline)
        rows = [{"image_id": 1, "eligible_gt": 1, "full_count": 20, "mask_count": 10,
                 "box_counts": [0] * 49 + [10] * 51}]
        self.record = {"schema_version": 1, "status": "calibrated", "split": "train",
                       "parameter_stage": "ImageNet initialization; no updates",
                       "initialization_sha256": IMAGENET_R50_SHA256,
                       "protocol_signature": protocol_signature(cfg, options),
                       "source_sha256": source_identity(ROOT),
                       "train_annotations_sha256": sha256(self.train),
                       "observations": rows, "fit": fit_count_calibration(rows)}
        self.calibration = self.directory / "calibration.json"
        self.write_record()
        values = asdict(options)
        values["reg_selection"] = "quality_iou_boxgate"
        self.method = {"name": "box_gate_test", "base_experiment": "base.yaml",
                       "box_gate_calibration": "calibration.json", "scda": values}
        self.path = self.directory / "method.yaml"
        self.write_method()

    def write_record(self):
        self.calibration.write_text(json.dumps(self.record))

    def write_method(self):
        self.path.write_text(yaml.safe_dump(self.method))

    def test_resolves_rho_and_embeds_evidence_in_fingerprinted_experiment(self):
        cfg = load_experiment(self.path)
        self.assertEqual(cfg["scda"]["box_gate_rho"], .5)
        self.assertEqual(cfg["box_gate_calibration"]["record"], self.record)
        self.assertEqual(cfg["box_gate_calibration"]["sha256"], sha256(self.calibration))
        before = fingerprint(cfg)
        cfg["box_gate_calibration"]["record"]["fit"]["rho"] = .6
        self.assertNotEqual(fingerprint(cfg), before)

    def test_missing_or_literal_rho_cannot_bypass_calibration(self):
        self.calibration.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "calibration missing"):
            load_experiment(self.path)
        self.method["scda"]["box_gate_rho"] = .5
        self.write_method()
        with self.assertRaisesRegex(ValueError, "not a YAML literal"):
            load_experiment(self.path)

    def test_changed_training_annotations_or_protocol_rejected(self):
        self.train.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, "annotations changed"):
            load_experiment(self.path)
        self.train.write_text("{}")
        baseline = yaml.safe_load(self.baseline.read_text())
        baseline["seed"] = 43
        self.baseline.write_text(yaml.safe_dump(baseline))
        with self.assertRaisesRegex(ValueError, "protocol differs"):
            load_experiment(self.path)

    def test_changed_source_or_coefficient_rejected(self):
        self.record["source_sha256"][next(iter(self.record["source_sha256"]))] = "changed"
        self.write_record()
        with self.assertRaisesRegex(ValueError, "code changed"):
            load_experiment(self.path)
        self.record["source_sha256"] = source_identity(ROOT)
        self.record["fit"]["rho"] = .6
        self.write_record()
        with self.assertRaisesRegex(ValueError, "rho/counts"):
            load_experiment(self.path)

    def test_val_or_trained_checkpoint_evidence_rejected(self):
        for key, value in (("split", "val"), ("parameter_stage", "trained checkpoint"),
                           ("initialization_sha256", "trained-checkpoint-hash")):
            original = self.record[key]
            self.record[key] = value
            self.write_record()
            with self.assertRaises(ValueError):
                load_experiment(self.path)
            self.record[key] = original


if __name__ == "__main__":
    unittest.main()
