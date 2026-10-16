"""Freeze actual reference settings without fabricating a new training recipe."""
import json
from pathlib import Path
import tempfile
import unittest

from methods.scda_proposal.experiment import load_experiment as load_scda
from methods.proposal_dcnv4.experiment import load_experiment
from scripts.prepare_quality_experiments import prepare

ROOT = Path(__file__).resolve().parents[3]


class QualityPreparationTests(unittest.TestCase):
    def test_preserves_reference_workers_schedule_and_data_but_binds_separate_gpus(self):
        reference = load_scda(ROOT / "methods/scda_proposal/configs/kvasir_freg_quality.yaml")
        reference["overrides"]["DATALOADER.NUM_WORKERS"] = 4
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory / "experiment.json"
            path.write_text(json.dumps(reference))
            output = prepare(path, directory / "frozen")
            plain = load_scda(output / "reference_quality.yaml")
            combined = load_experiment(output / "dcn_quality.yaml")
            for cfg in (plain, combined):
                for key in ("overrides", "dataset", "model", "training_budget", "seed", "evaluation"):
                    self.assertEqual(cfg[key], reference[key])
            self.assertEqual(combined["runtime"]["visible_gpus"], "0")
            # Box config deliberately cannot resolve without a real artifact.
            with self.assertRaises(FileNotFoundError):
                load_scda(output / "box_gated_iou.yaml")
            import yaml
            box_base = yaml.safe_load((output / "box_base.yaml").read_text())
            self.assertEqual(box_base["runtime"]["visible_gpus"], "1")
            self.assertFalse(json.loads((output / "preparation.json").read_text())["training_started"])
            with self.assertRaises(FileExistsError):
                prepare(path, output)

    def test_rejects_old_policy_seed_or_batch_instead_of_silently_migrating_it(self):
        reference = load_scda(ROOT / "methods/scda_proposal/configs/kvasir_freg_quality.yaml")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.json"
            for mutation in ("policy", "batch", "seed"):
                candidate = json.loads(json.dumps(reference))
                if mutation == "policy": candidate["scda"]["reg_selection"] = "core_plus_transition"
                if mutation == "batch": candidate["overrides"]["SOLVER.IMS_PER_BATCH"] = 4
                if mutation == "seed": candidate["seed"] = 43
                path.write_text(json.dumps(candidate))
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    prepare(path, Path(directory) / mutation)


if __name__ == "__main__":
    unittest.main()
