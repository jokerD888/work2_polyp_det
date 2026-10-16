"""Calibration identity is required, bound to train data and copied into cfg."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import yaml
from PIL import Image

from methods.scda_proposal.config import SCDAOptions
from methods.scda_proposal.experiment import load_experiment
from methods.scda_proposal.box_gate_calibration import (fit_count_calibration,
                    CALIBRATION_SCHEMA_VERSION, IMAGENET_R50_SHA256,
                    native_config_identity, training_images_identity,
                    protocol_signature, source_identity)
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
        self.train_contents = json.dumps({"images": [{"id": 1, "file_name": "one.png",
                                                       "width": 8, "height": 8}]})
        self.train.write_text(self.train_contents)
        self.image_root = self.directory / "images"
        self.image_root.mkdir()
        self.image = self.image_root / "one.png"
        Image.new("RGB", (8, 8), (10, 20, 30)).save(self.image)
        baseline["dataset"]["train"]["annotations"] = str(self.train)
        baseline["dataset"]["train"]["image_root"] = str(self.image_root)
        # A transitive local model config chain, independent of the native checkout.
        self.model_root = self.directory / "model.yaml"
        self.model_base = self.directory / "base_model.yaml"
        self.model_grandparent = self.directory / "defaults.yaml"
        self.model_root.write_text("_BASE_: base_model.yaml\nMODEL:\n  NUM_OBJECT_QUERIES: 300\n")
        self.model_base.write_text("_BASE_: defaults.yaml\nMODEL:\n  WIDTH: 256\n")
        self.model_grandparent.write_text("MODEL:\n  DEPTH: 50\n")
        baseline["model"]["config"] = str(self.model_root)
        self.baseline = self.directory / "base.yaml"
        self.baseline.write_text(yaml.safe_dump(baseline))
        options = SCDAOptions(enabled=True, reg_selection="quality_iou", lambda_cls=0.)
        cfg = load_config(self.baseline)
        rows = [{"image_id": 1, "eligible_gt": 1, "full_count": 20, "mask_count": 10,
                 "box_counts": [0] * 49 + [10] * 51}]
        self.record = {"schema_version": CALIBRATION_SCHEMA_VERSION,
                       "status": "calibrated", "split": "train",
                       "parameter_stage": "ImageNet initialization; no updates",
                       "initialization_sha256": IMAGENET_R50_SHA256,
                       "protocol_signature": protocol_signature(cfg, options),
                       "source_sha256": source_identity(ROOT),
                       "train_annotations_sha256": sha256(self.train),
                       "samples": 1, "image_ids": [1],
                       "native_model_config": native_config_identity(cfg["model"]["config"]),
                       "training_images": training_images_identity(cfg, [1]),
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
        self.train.write_text(self.train_contents)
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

    def test_same_dimensions_replaced_image_rejected_without_annotation_change(self):
        annotation_hash = sha256(self.train)
        original_image_hash = sha256(self.image)
        Image.new("RGB", (8, 8), (30, 20, 10)).save(self.image)
        with Image.open(self.image) as image:
            self.assertEqual(image.size, (8, 8))
        self.assertEqual(sha256(self.train), annotation_hash)
        self.assertNotEqual(sha256(self.image), original_image_hash)
        with self.assertRaisesRegex(ValueError, "training images changed"):
            load_experiment(self.path)

    def test_entry_direct_and_transitive_model_edits_rejected_at_same_paths(self):
        for path, old, new in ((self.model_root, "300", "900"),
                               (self.model_base, "256", "512"),
                               (self.model_grandparent, "50", "101")):
            with self.subTest(path=path.name):
                contents = path.read_text()
                path.write_text(contents.replace(old, new))
                try:
                    with self.assertRaisesRegex(ValueError, "model YAML or _BASE_ contents changed"):
                        load_experiment(self.path)
                finally:
                    path.write_text(contents)
        self.assertEqual(load_experiment(self.path)["scda"]["box_gate_rho"], .5)

    def test_deleted_model_dependency_rejected(self):
        self.model_grandparent.unlink()
        with self.assertRaises(FileNotFoundError):
            load_experiment(self.path)

    def test_config_graph_handles_shared_multiple_bases_and_python_yaml_tags(self):
        second = self.directory / "second.yaml"
        second.write_text("_BASE_: defaults.yaml\nPYTHON_VALUE: !!python/tuple [1, 2]\n")
        self.model_root.write_text("_BASE_: [base_model.yaml, second.yaml]\n")
        identity = native_config_identity(self.model_root)
        expected = (self.model_root, self.model_base, self.model_grandparent, second)
        self.assertEqual(identity["files_sha256"], {str(p.resolve()): sha256(p) for p in expected})

    def test_model_dependency_cycle_rejected(self):
        self.model_grandparent.write_text("_BASE_: model.yaml\n")
        with self.assertRaisesRegex(ValueError, "Cyclic"):
            native_config_identity(self.model_root)

    def test_schema_without_content_hashes_requires_real_recalibration(self):
        self.record["schema_version"] = 1
        self.write_record()
        with self.assertRaisesRegex(ValueError, "schema.*regenerate"):
            load_experiment(self.path)

    def test_sample_ids_and_counts_must_match_observations(self):
        for key, value in (("samples", 2), ("image_ids", [2])):
            with self.subTest(key=key):
                original = self.record[key]
                self.record[key] = value
                self.write_record()
                with self.assertRaisesRegex(ValueError, "sample IDs/counts"):
                    load_experiment(self.path)
                self.record[key] = original

    def test_train_identity_never_reads_validation_files(self):
        baseline = yaml.safe_load(self.baseline.read_text())
        baseline["dataset"]["val"]["annotations"] = str(self.directory / "missing_val.json")
        baseline["dataset"]["val"]["image_root"] = str(self.directory / "missing_val_images")
        self.baseline.write_text(yaml.safe_dump(baseline))
        options = SCDAOptions(enabled=True, reg_selection="quality_iou", lambda_cls=0.)
        self.record["protocol_signature"] = protocol_signature(load_config(self.baseline), options)
        self.write_record()
        self.assertEqual(load_experiment(self.path)["scda"]["box_gate_rho"], .5)

    def test_invalid_sample_ids_and_escaping_image_paths_rejected(self):
        cfg = load_config(self.baseline)
        for ids in ([], [1, 1], [2], [True]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                training_images_identity(cfg, ids)
        self.train.write_text(json.dumps({"images": [{"id": 1, "file_name": "../outside.png"}]}))
        with self.assertRaisesRegex(ValueError, "escapes"):
            training_images_identity(cfg, [1])


if __name__ == "__main__":
    unittest.main()
