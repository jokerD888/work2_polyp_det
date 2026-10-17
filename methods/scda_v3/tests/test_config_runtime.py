from dataclasses import asdict
from pathlib import Path
import copy
import tempfile
import json
import unittest
import yaml
from methods.scda_v3.config import V3Options
from methods.scda_v3.experiment import load_experiment
from methods.scda_v3.run import adapter_sources, command

ROOT = Path(__file__).resolve().parents[3]


class ConfigRuntimeTests(unittest.TestCase):
    def test_v3a_and_v3b_have_identical_protocol_head_and_loss_only_assignment_differs(self):
        a, b = (load_experiment(ROOT/f"methods/scda_v3/configs/{name}.yaml") for name in ("v3a", "v3b"))
        self.assertEqual(a["overrides"]["SOLVER.BASE_LR"], .0001)
        self.assertEqual(a["overrides"]["DATALOADER.NUM_WORKERS"], 4)
        self.assertEqual(a["overrides"]["SOLVER.STEPS"], [4400, 4950])
        self.assertEqual(a["training_budget"]["epochs"], 100)
        for key in ("dataset", "model", "runtime", "overrides", "training_budget", "seed", "evaluation"):
            self.assertEqual(a[key], b[key], key)
        av, bv = copy.deepcopy(a["v3"]), copy.deepcopy(b["v3"])
        av.pop("assignment_policy"); bv.pop("assignment_policy")
        self.assertEqual(av, bv)
        self.assertEqual(a["native_model_config"], b["native_model_config"])

    def test_worker_and_snapshot_include_new_and_reused_code_without_legacy_run_mutation(self):
        cfg = load_experiment(ROOT/"methods/scda_v3/configs/v3a.yaml")
        argv = command(cfg, "snapshot.json", "output", "train")
        self.assertIn("methods.scda_v3.worker", argv)
        sources = adapter_sources(cfg)
        for name in ("methods/scda_v3/model.py", "methods/scda_v3/decoder.py",
                     "methods/scda_proposal/scda_assigner.py", "methods/scda_proposal/mapper.py",
                     "methods/scda_v3/configs/base_kvasir.yaml"):
            self.assertIn(name, sources)

    def test_rejects_accidental_high_lr_wrong_workers_or_mature_initialization(self):
        base = yaml.safe_load((ROOT/"methods/scda_v3/configs/base_kvasir.yaml").read_text())
        base["project_root"] = str(ROOT)
        method = {"name": "test_v3", "base_experiment": "base.yaml", "v3": asdict(V3Options())}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory/"method.yaml").write_text(yaml.safe_dump(method))
            for key, value in (("SOLVER.BASE_LR", .0004), ("DATALOADER.NUM_WORKERS", 2),
                               ("SOLVER.STEPS", [4889, 5296])):
                edited = copy.deepcopy(base); edited["overrides"][key] = value
                (directory/"base.yaml").write_text(yaml.safe_dump(edited))
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "low-LR"):
                    load_experiment(directory/"method.yaml")
            base["model"]["weights"] = "runs/mature/model_best.pth"
            (directory/"base.yaml").write_text(yaml.safe_dump(base))
            with self.assertRaisesRegex(ValueError, "ImageNet R50"):
                load_experiment(directory/"method.yaml")

    def test_prepare_uses_actual_reference_paths_and_preserves_protocol(self):
        from methods.scda_v3.prepare import prepare
        from polypbench.config import load_config
        cfg = load_config(ROOT/"methods/scda_v3/configs/base_kvasir.yaml")
        cfg["scda"] = asdict(V3Options().legacy())
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            reference = directory/"experiment.json"; reference.write_text(json.dumps(cfg))
            output = prepare(reference, directory/"prepared")
            a, b = load_experiment(output/"v3a.yaml"), load_experiment(output/"v3b.yaml")
            for method in (a, b):
                for key in ("dataset", "model", "overrides", "seed", "training_budget"):
                    self.assertEqual(method[key], cfg[key])
            self.assertEqual(a["runtime"]["visible_gpus"], "0")
            self.assertEqual(b["runtime"]["visible_gpus"], "1")
            with self.assertRaises(FileExistsError): prepare(reference, output)


if __name__ == "__main__": unittest.main()
