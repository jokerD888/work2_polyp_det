"""Real ordinary-Conv CPU checks and operator/provenance regression tests."""
from dataclasses import asdict
import gc
import importlib.abc
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

import torch

from methods.proposal_dcnv4.config import DCNv4Options, add_dcnv4_config
from methods.proposal_dcnv4.conv3x3_adapter import Conv3x3Residual
from methods.proposal_dcnv4.experiment import load_experiment
from methods.proposal_dcnv4.tests.test_cpu import ROOT, PixelFixture, PredictorFixture
from polypbench.config import fingerprint
from polypbench.io import write_json


class RejectDCNv4Import(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "DCNv4" or fullname.startswith("DCNv4."):
            raise AssertionError("Ordinary Conv attempted to import official DCNv4")
        return None


class ConvControlContracts(unittest.TestCase):
    def options(self, **overrides):
        return DCNv4Options(operator="conv3x3", **overrides).validate()

    def test_operator_defaults_validation_and_legacy_yaml(self):
        self.assertEqual(DCNv4Options().operator, "dcnv4")
        self.assertFalse(DCNv4Options().enabled)
        with self.assertRaises(ValueError):
            DCNv4Options(operator="dcnv2").validate()
        # Ordinary Conv is not constrained by DCNv4's group channel kernel.
        self.options(channels=32)
        import yaml
        path = ROOT / "methods/proposal_dcnv4/configs/dcn_only.yaml"
        legacy = yaml.safe_load(path.read_text())
        legacy["base_experiment"] = str(ROOT / "configs/maskdino_kvasir_official.yaml")
        legacy["dcnv4"].pop("operator")
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory) / "legacy.yaml"
            old.write_text(yaml.safe_dump(legacy))
            self.assertEqual(load_experiment(old)["dcnv4"]["operator"], "dcnv4")

    def test_real_conv_count_zero_initialization_first_step_and_rng(self):
        before = torch.get_rng_state().clone()
        adapter = Conv3x3Residual(self.options())
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in adapter.parameters()), 590080)
        self.assertEqual(adapter.conv.groups, 1)
        self.assertTrue(torch.equal(adapter.conv.weight, torch.zeros_like(adapter.conv.weight)))
        self.assertTrue(torch.equal(adapter.conv.bias, torch.zeros_like(adapter.conv.bias)))
        self.assertNotIn("alpha", dict(adapter.named_parameters()))
        x = torch.randn(2, 256, 5, 7).transpose(2, 3).detach().requires_grad_()
        y = adapter(x)
        self.assertTrue(torch.equal(x, y))
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(adapter.conv.weight.grad).all())
        self.assertGreater(adapter.conv.weight.grad.norm().item(), 0)
        self.assertGreater(adapter.conv.bias.grad.norm().item(), 0)
        torch.testing.assert_close(x.grad, 2 * x.detach() / x.numel())
        with torch.no_grad():
            adapter.conv.weight.add_(adapter.conv.weight.grad, alpha=-.1)
        self.assertFalse(torch.equal(adapter(x), x))

    def test_conv_fp32_under_autocast_and_alpha_zero_bypass(self):
        adapter = Conv3x3Residual(self.options())
        seen = []
        handle = adapter.conv.register_forward_pre_hook(
            lambda module, args: seen.append((args[0].dtype, torch.is_autocast_enabled("cpu"))))
        x = torch.randn(1, 256, 3, 5).bfloat16()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            y = adapter(x)
        handle.remove()
        self.assertEqual(seen, [(torch.float32, False)])
        self.assertEqual(y.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(x, y))
        zero = Conv3x3Residual(self.options(alpha=0))
        with patch.object(zero.conv, "forward", side_effect=AssertionError("must bypass")):
            self.assertIs(zero(x), x)
        with self.assertRaises(TypeError):
            adapter.half()(torch.randn(1, 256, 3, 5))

    def test_four_presets_inherit_identical_data_and_training_recipe(self):
        values = {name: load_experiment(ROOT / f"methods/proposal_dcnv4/configs/{name}.yaml")
                  for name in ("dcn_only", "dcn_fv2", "conv_only", "conv_fv2")}
        for name, value in values.items():
            for key in ("overrides", "model", "dataset", "training_budget", "seed", "runtime"):
                self.assertEqual(value[key], values["dcn_only"][key], (name, key))
            self.assertEqual(value["dcnv4"]["operator"], "conv3x3" if name.startswith("conv") else "dcnv4")
            self.assertEqual(value["scda"]["enabled"], name.endswith("fv2"))
            self.assertEqual(value["dcnv4"]["alpha"], .1)
        self.assertEqual(values["conv_fv2"]["scda"], values["dcn_fv2"]["scda"])

    def test_cli_override_in_effective_configuration_and_fingerprint(self):
        path = ROOT / "methods/proposal_dcnv4/configs/dcn_only.yaml"
        native = load_experiment(path)
        conv = load_experiment(path, operator="conv3x3")
        self.assertEqual(conv["dcnv4"]["operator"], "conv3x3")
        self.assertEqual(conv["operator_selection"], {
            "configured": "dcnv4", "effective": "conv3x3", "source": "command_line"})
        self.assertNotEqual(fingerprint(native), fingerprint(conv))
        self.assertEqual(fingerprint(conv), fingerprint(load_experiment(path, operator="conv3x3")))
        import methods.proposal_dcnv4.run as runner
        argv = ["proposal", "--config", str(path), "--operator", "conv3x3",
                "--mode", "train", "--run-id", "unit-test-no-training"]
        with patch.object(sys, "argv", argv), patch.object(runner, "run") as launch:
            runner.main()
        self.assertEqual(launch.call_args.args[0]["dcnv4"]["operator"], "conv3x3")

    def test_prediction_operator_mismatch_rejected_before_worker(self):
        import methods.proposal_dcnv4.run as runner
        path = ROOT / "methods/proposal_dcnv4/configs/dcn_only.yaml"
        train = load_experiment(path, operator="conv3x3")
        with tempfile.TemporaryDirectory() as directory:
            train["output_root"] = directory
            run_root = Path(directory) / "test-only"
            run_root.mkdir()
            write_json(run_root / "manifest.json", {
                "adapter_operator": "conv3x3", "config_fingerprint": fingerprint(train),
                "dataset": {}, "repo": {}})
            wrong = load_experiment(path)  # accidentally omitted --operator
            wrong["output_root"] = directory
            with patch.object(runner, "verify_repo", return_value={}), \
                 patch.object(runner, "validate_dataset", return_value={}), \
                 patch.object(runner, "run_process") as process:
                with self.assertRaisesRegex(ValueError, "Operator differs from training"):
                    runner.run(wrong, "predict", run_id="test-only")
                process.assert_not_called()
            self.assertFalse((run_root / "predict_val").exists())

    def test_conv_head_executes_real_conv_on_only_selected_scale(self):
        from detectron2.layers import ShapeSpec
        from methods.proposal_dcnv4.dcnv4_head import ProposalDCNv4Head
        pixel, predictor = PixelFixture(), PredictorFixture()
        with patch("methods.proposal_dcnv4.dcnv4_head.DCNv4Residual",
                   side_effect=AssertionError("Conv must not construct DCNv4")):
            head = ProposalDCNv4Head({"res3": ShapeSpec(channels=512, stride=8)},
                                    num_classes=1, pixel_decoder=pixel, transformer_predictor=predictor,
                                    dcnv4_options=self.options(enabled=True))
        with torch.no_grad():
            head.dcnv4_adapter.conv.bias.fill_(2)
        features = {"res3": torch.randn(2, 512, 8, 12)}
        for training in (True, False):
            head.train(training)
            head(features)
            levels, mask_feature, _, _ = predictor.received
            torch.testing.assert_close(levels[0], pixel.levels[0] + .2)
            self.assertIs(mask_feature, pixel.mask_feature)
            for index in (1, 2, 3):
                self.assertIs(levels[index], pixel.levels[index])

    def test_real_conv_full_model_and_optimizer_need_no_dcnv4(self):
        from detectron2.config import get_cfg
        from detectron2.modeling import build_model
        from detectron2.projects.deeplab import add_deeplab_config
        from methods.proposal_dcnv4.integration import configure_model, register
        from methods.scda_proposal.config import add_scda_config
        import train_net
        register(ROOT / "third_party/MaskDINO")

        def config(enabled, scda):
            cfg = get_cfg()
            add_deeplab_config(cfg)
            train_net.add_maskdino_config(cfg)
            add_scda_config(cfg)
            add_dcnv4_config(cfg)
            cfg.merge_from_file(str(ROOT / "third_party/MaskDINO/configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml"))
            cfg.MODEL.DEVICE = "cpu"
            cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
            cfg.SOLVER.OPTIMIZER = "ADAMW"
            cfg.MODEL.PROPOSAL_DCNV4.ENABLED = enabled
            cfg.MODEL.PROPOSAL_DCNV4.OPERATOR = "conv3x3"
            cfg.MODEL.SCDA.ENABLED = scda
            configure_model(cfg)
            return cfg

        blocker = RejectDCNv4Import()
        sys.meta_path.insert(0, blocker)
        try:
            torch.manual_seed(42)
            with redirect_stdout(io.StringIO()):
                model = build_model(config(False, False))
            reference = {key: value.clone() for key, value in model.state_dict().items()}
            native_count = sum(p.numel() for p in model.parameters())
            del model
            for scda in (False, True):
                cfg = config(True, scda)
                torch.manual_seed(42)
                with redirect_stdout(io.StringIO()):
                    model = build_model(cfg)
                common = {key: value for key, value in model.state_dict().items()
                          if ".dcnv4_adapter." not in key}
                self.assertEqual(common.keys(), reference.keys())
                self.assertTrue(all(torch.equal(reference[key], value) for key, value in common.items()))
                self.assertEqual(sum(p.numel() for p in model.parameters()) - native_count, 590080)
                optimizer = train_net.Trainer.build_optimizer(cfg, model)
                identities = [id(p) for group in optimizer.param_groups for p in group["params"]]
                self.assertEqual(set(identities), {id(p) for p in model.parameters() if p.requires_grad})
                self.assertEqual(len(identities), len(set(identities)))
                del model, optimizer
                gc.collect()
        finally:
            sys.meta_path.remove(blocker)
        self.assertFalse(any(name == "DCNv4" or name.startswith("DCNv4.") for name in sys.modules))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
