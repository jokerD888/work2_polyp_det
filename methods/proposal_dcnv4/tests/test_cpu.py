"""CPU-only contract checks. Injected operators are NOT DCNv4 accuracy tests."""
from dataclasses import asdict, replace
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

from methods.proposal_dcnv4.config import DCNv4Options, add_dcnv4_config
from methods.proposal_dcnv4.dcnv4_adapter import DCNv4Residual
from methods.proposal_dcnv4.experiment import load_experiment

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "third_party/MaskDINO"))
torch.set_num_threads(2)


class TokenProbe(nn.Module):
    """Explicit test double for shape/routing/gradient, no dynamic sampling."""
    def __init__(self, channels, **kwargs):
        super().__init__()
        self.gain = nn.Parameter(torch.full((channels,), .5))
        self.calls = 0
        self.options = kwargs

    def forward(self, tokens, shape=None):
        self.calls += 1
        self.seen = (tokens.detach().clone(), shape, torch.is_autocast_enabled("cpu"))
        return tokens * self.gain


class PixelFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_feature_strides = [32, 16, 8]
        self.total_num_feature_levels = 4
        self.low_resolution_index = 0
        self.feature_order = "low2high"
        self.levels = [torch.randn(2, 256, h, w) for h, w in ((8, 12), (4, 6), (2, 3), (1, 2))]
        self.mask_feature = torch.randn(2, 256, 16, 24)

    def forward_features(self, features, mask):
        self.received = (features, mask)
        return self.mask_feature, self.levels[0], self.levels


class PredictorFixture(nn.Module):
    def forward(self, levels, mask_feature, mask, targets=None):
        self.received = (levels, mask_feature, mask, targets)
        return sum(level.square().mean() for level in levels)


class CPUContracts(unittest.TestCase):
    def test_invalid_options(self):
        for values in ({"channels": 255}, {"group": 3}, {"alpha": float("nan")},
                       {"enabled": 1}, {"force_fp32": False}, {"kernel_size": 5}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                DCNv4Options(**values).validate()

    def test_defaults_are_fixed_alpha_and_off(self):
        options = DCNv4Options().validate()
        self.assertFalse(options.enabled)
        self.assertEqual(options.alpha, .1)
        self.assertEqual(options.feature_stride, 8)

    def test_presets_inherit_untouched_baseline_and_fv2(self):
        from polypbench.config import load_config
        from methods.scda_proposal.experiment import load_experiment as load_f
        baseline = load_config(ROOT / "configs/maskdino_kvasir_official.yaml")
        f = load_f(ROOT / "methods/scda_proposal/configs/kvasir_v2.yaml")
        d = load_experiment(ROOT / "methods/proposal_dcnv4/configs/dcn_only.yaml")
        df = load_experiment(ROOT / "methods/proposal_dcnv4/configs/dcn_fv2.yaml")
        for key in ("overrides", "dataset", "model", "runtime", "training_budget", "seed"):
            self.assertEqual(d[key], baseline[key], key)
            self.assertEqual(df[key], baseline[key], key)
        self.assertEqual(df["scda"], f["scda"])
        self.assertFalse(d["scda"]["enabled"])
        self.assertEqual(df["scda"]["reg_selection"], "core_plus_transition")

    def test_layout_and_residual_rectangular_noncontiguous(self):
        x = torch.randn(2, 256, 5, 7).transpose(2, 3)
        adapter = DCNv4Residual(DCNv4Options(), operator_factory=TokenProbe)
        result = adapter(x)
        torch.testing.assert_close(result, x * 1.05)
        tokens, shape, _ = adapter.dcn.seen
        self.assertEqual(shape, (7, 5))
        torch.testing.assert_close(tokens.reshape(2, 7, 5, 256).permute(0, 3, 1, 2), x)
        self.assertTrue(tokens.is_contiguous())

    def test_gradient_to_feature_and_operator(self):
        x = torch.randn(2, 256, 3, 5, requires_grad=True)
        adapter = DCNv4Residual(DCNv4Options(), operator_factory=TokenProbe)
        adapter(x).square().sum().backward()
        self.assertGreater(x.grad.norm().item(), 0)
        self.assertGreater(adapter.dcn.gain.grad.norm().item(), 0)
        self.assertNotIn("alpha", dict(adapter.named_parameters()))
        self.assertNotIn("alpha", dict(adapter.named_buffers()))

    def test_alpha_zero_does_not_execute_operator(self):
        adapter = DCNv4Residual(DCNv4Options(alpha=0), operator_factory=TokenProbe)
        x = torch.randn(1, 256, 3, 5, requires_grad=True)
        y = adapter(x)
        self.assertIs(y, x)
        self.assertEqual(adapter.dcn.calls, 0)
        y.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.ones_like(x)))

    def test_adapter_initialization_preserves_cpu_rng(self):
        class RandomInitializer(TokenProbe):
            def __init__(self, channels, **kwargs):
                super().__init__(channels, **kwargs)
                self.proj = nn.Linear(channels, channels)
        before = torch.get_rng_state().clone()
        DCNv4Residual(DCNv4Options(), operator_factory=RandomInitializer)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_cpu_autocast_does_not_cast_operator(self):
        adapter = DCNv4Residual(DCNv4Options(), operator_factory=TokenProbe)
        x = torch.randn(1, 256, 3, 5).bfloat16()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = adapter(x)
        tokens, _, amp = adapter.dcn.seen
        self.assertEqual(tokens.dtype, torch.float32)
        self.assertFalse(amp)
        self.assertEqual(result.dtype, x.dtype)

    def test_wrong_shape_rejected(self):
        adapter = DCNv4Residual(DCNv4Options(), operator_factory=TokenProbe)
        for x in (torch.zeros(1, 255, 3, 5), torch.zeros(1, 256, 15)):
            with self.assertRaises(ValueError):
                adapter(x)

    def test_half_parameters_rejected(self):
        adapter = DCNv4Residual(DCNv4Options(), operator_factory=TokenProbe).half()
        with self.assertRaises(TypeError):
            adapter(torch.randn(1, 256, 3, 5))

    def test_registration_does_not_import_cuda_extension(self):
        from methods.proposal_dcnv4.integration import register
        register(ROOT / "third_party/MaskDINO")
        self.assertNotIn("DCNv4.ext", sys.modules)

    def test_feature_order_matches_native_metadata(self):
        from methods.proposal_dcnv4.dcnv4_head import feature_strides
        pixel = PixelFixture()
        self.assertEqual(feature_strides(pixel), [8, 16, 32, 64])
        pixel.feature_order = "high2low"
        pixel.transformer_feature_strides = [8, 16, 32]
        pixel.low_resolution_index = 2
        self.assertEqual(feature_strides(pixel), [64, 32, 16, 8])

    def test_real_derived_head_routes_one_level_and_preserves_args(self):
        from detectron2.layers import ShapeSpec
        from methods.proposal_dcnv4.dcnv4_head import ProposalDCNv4Head
        pixel, predictor = PixelFixture(), PredictorFixture()
        adapter = DCNv4Residual(DCNv4Options(), operator_factory=TokenProbe)
        head = ProposalDCNv4Head({"res3": ShapeSpec(channels=512, stride=8)},
                                num_classes=1, pixel_decoder=pixel,
                                transformer_predictor=predictor,
                                dcnv4_options=DCNv4Options(), adapter=adapter)
        features = {"res3": torch.randn(2, 512, 8, 12)}
        mask, targets = object(), object()
        for training in (True, False):
            head.train(training)
            head(features, mask, targets)
            levels, mf, seen_mask, seen_targets = predictor.received
            self.assertIs(mf, pixel.mask_feature)
            self.assertIs(seen_mask, mask)
            self.assertIs(seen_targets, targets)
            torch.testing.assert_close(levels[0], pixel.levels[0] * 1.05)
            for index in range(1, 4):
                self.assertIs(levels[index], pixel.levels[index])
            self.assertIsNot(levels, pixel.levels)
        self.assertEqual(adapter.dcn.calls, 2)  # deployed module is NOT training-only

    def test_alpha_zero_head_matches_native_output_and_gradients(self):
        from detectron2.layers import ShapeSpec
        from maskdino.modeling.meta_arch.maskdino_head import MaskDINOHead
        from methods.proposal_dcnv4.dcnv4_head import ProposalDCNv4Head
        pixel, predictor = PixelFixture(), PredictorFixture()
        pixel.levels = [x.requires_grad_() for x in pixel.levels]
        native = MaskDINOHead({"res3": ShapeSpec(channels=512, stride=8)},
                              num_classes=1, pixel_decoder=pixel, transformer_predictor=predictor)
        enhanced = ProposalDCNv4Head({"res3": ShapeSpec(channels=512, stride=8)},
                                    num_classes=1, pixel_decoder=pixel, transformer_predictor=predictor,
                                    dcnv4_options=DCNv4Options(alpha=0),
                                    adapter=DCNv4Residual(DCNv4Options(alpha=0), operator_factory=TokenProbe))
        features = {"res3": torch.randn(2, 512, 8, 12)}
        a, b = native(features), enhanced(features)
        self.assertTrue(torch.equal(a, b))
        ga = torch.autograd.grad(a, pixel.levels)
        gb = torch.autograd.grad(b, pixel.levels)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(ga, gb)))

    def test_config_four_independent_switches(self):
        import train_net
        from detectron2.config import get_cfg
        from detectron2.projects.deeplab import add_deeplab_config
        from methods.scda_proposal.config import add_scda_config
        from methods.proposal_dcnv4.integration import configure_model
        for dcn in (False, True):
            for scda in (False, True):
                cfg = get_cfg()
                add_deeplab_config(cfg)
                train_net.add_maskdino_config(cfg)
                add_scda_config(cfg)
                add_dcnv4_config(cfg)
                cfg.merge_from_file(str(ROOT / "third_party/MaskDINO/configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml"))
                cfg.MODEL.PROPOSAL_DCNV4.ENABLED = dcn
                cfg.MODEL.SCDA.ENABLED = scda
                configure_model(cfg)
                self.assertEqual(cfg.MODEL.SEM_SEG_HEAD.NAME,
                                 "ProposalDCNv4Head" if dcn else "MaskDINOHead")
                self.assertEqual(cfg.MODEL.MaskDINO.TRANSFORMER_DECODER_NAME,
                                 "SCDAProposalDecoder" if scda else "MaskDINODecoder")

    def test_missing_extension_fails_without_conv_fallback(self):
        with patch("methods.proposal_dcnv4.backend.importlib.import_module", side_effect=ImportError("absent")):
            with self.assertRaisesRegex(RuntimeError, "No CPU/Conv/DCNv2 fallback"):
                DCNv4Residual(DCNv4Options())

    def test_full_cpu_model_construction_preserves_native_initialization(self):
        # Real MaskDINO / SCDA model, pixel decoder, decoder and optimizer.
        # Only the unavailable DCNv4 CUDA operator is an explicit test double.
        import gc
        import train_net
        from detectron2.config import get_cfg
        from detectron2.modeling import build_model
        from detectron2.projects.deeplab import add_deeplab_config
        from methods.scda_proposal.config import add_scda_config
        from methods.proposal_dcnv4.integration import configure_model, register
        register(ROOT / "third_party/MaskDINO")

        def config(dcn, scda):
            cfg = get_cfg()
            add_deeplab_config(cfg)
            train_net.add_maskdino_config(cfg)
            add_scda_config(cfg)
            add_dcnv4_config(cfg)
            native_path = ROOT / "third_party/MaskDINO/configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml"
            cfg.merge_from_file(str(native_path))
            cfg.MODEL.DEVICE = "cpu"
            cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
            cfg.MODEL.PROPOSAL_DCNV4.ENABLED = dcn
            cfg.MODEL.SCDA.ENABLED = scda
            cfg.SOLVER.OPTIMIZER = "ADAMW"
            configure_model(cfg)
            return cfg

        torch.manual_seed(42)
        native = build_model(config(False, False))
        reference = {key: value.clone() for key, value in native.state_dict().items()}
        del native
        for scda in (False, True):
            cfg = config(True, scda)
            torch.manual_seed(42)
            with patch("methods.proposal_dcnv4.dcnv4_head.DCNv4Residual",
                       side_effect=lambda options: DCNv4Residual(options, operator_factory=TokenProbe)):
                model = build_model(cfg)
            common = {key: value for key, value in model.state_dict().items()
                      if ".dcnv4_adapter." not in key}
            self.assertEqual(reference.keys(), common.keys())
            self.assertTrue(all(torch.equal(reference[key], value) for key, value in common.items()))
            self.assertEqual(model.sem_seg_head.dcnv4_feature_strides, (8, 16, 32, 64))
            optimizer = train_net.Trainer.build_optimizer(cfg, model)
            actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
            expected = {id(p) for p in model.parameters() if p.requires_grad}
            self.assertEqual(len(actual), len(set(actual)))
            self.assertEqual(set(actual), expected)
            del optimizer, model
            gc.collect()
        self.assertFalse(torch.cuda.is_initialized())

    def test_no_cuda_context_created(self):
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
