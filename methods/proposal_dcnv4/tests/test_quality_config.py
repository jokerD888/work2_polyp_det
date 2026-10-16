"""Configuration contracts; no torch, CUDA or dataset access required."""
import copy
from pathlib import Path
import tempfile
import unittest

import yaml

from methods.proposal_dcnv4.experiment import load_experiment
from methods.scda_proposal.experiment import load_experiment as load_scda
from polypbench.config import load_config


ROOT = Path(__file__).resolve().parents[3]
QUALITY = ROOT / "methods/scda_proposal/configs/kvasir_freg_quality.yaml"
COMBINED = ROOT / "methods/proposal_dcnv4/configs/dcn_freg_quality.yaml"


class QualityConfigTests(unittest.TestCase):
    def test_quality_inherits_complete_reported_batch16_protocol(self):
        config = load_scda(QUALITY)
        self.assertEqual(config["training_budget"], {
            "train_images": 880, "iterations_per_epoch": 55, "epochs": 100})
        expected = {"SOLVER.IMS_PER_BATCH": 16, "SOLVER.MAX_ITER": 5500,
                    "SOLVER.BASE_LR": 0.0004, "SOLVER.WARMUP_ITERS": 125,
                    "SOLVER.STEPS": [4400, 4950], "SOLVER.CHECKPOINT_PERIOD": 275,
                    "TEST.EVAL_PERIOD": 275}
        for key, value in expected.items():
            self.assertEqual(config["overrides"][key], value)
        self.assertEqual(config["scda"]["reg_selection"], "quality_iou")
        self.assertEqual(config["scda"]["lambda_cls"], 0.0)
        self.assertEqual(config["scda"]["lambda_box"], 0.05)

    def test_combined_changes_only_adapter_not_scda_or_protocol(self):
        plain, combined = load_scda(QUALITY), load_experiment(COMBINED)
        for key in ("scda", "dataset", "model", "runtime", "overrides",
                    "training_budget", "seed", "evaluation", "checkpoint_selection"):
            self.assertEqual(plain[key], combined[key], key)
        self.assertTrue(combined["dcnv4"]["enabled"])
        self.assertEqual(combined["dcnv4"]["operator"], "dcnv4")
        self.assertEqual(combined["dcnv4"]["alpha"], 0.1)

    def test_legacy_batch4_and_v2_presets_remain_valid(self):
        baseline = load_config(ROOT / "configs/maskdino_kvasir_official.yaml")
        self.assertEqual(baseline["overrides"]["SOLVER.IMS_PER_BATCH"], 4)
        old = load_experiment(ROOT / "methods/proposal_dcnv4/configs/dcn_fv2.yaml")
        self.assertEqual(old["scda"]["reg_selection"], "core_plus_transition")
        self.assertEqual(old["scda"]["lambda_cls"], 0.05)

    def test_combined_rejects_undeclared_scda_variants(self):
        method = yaml.safe_load(QUALITY.read_text())
        adapter = yaml.safe_load(COMBINED.read_text())
        method["base_experiment"] = str(ROOT / "configs/maskdino_kvasir_official_b16.yaml")
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            adapter["base_experiment"] = "method.yaml"
            (directory / "adapter.yaml").write_text(yaml.safe_dump(adapter))
            for changes in ({"reg_selection": "quality_iou_boxonly"},
                            {"lambda_cls": 0.05}, {"lambda_box": 0.1}, {"enabled": False}):
                candidate = copy.deepcopy(method)
                candidate["scda"].update(changes)
                (directory / "method.yaml").write_text(yaml.safe_dump(candidate))
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    load_experiment(directory / "adapter.yaml")


if __name__ == "__main__":
    unittest.main()
