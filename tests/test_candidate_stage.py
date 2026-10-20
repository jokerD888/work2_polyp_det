import sys
import tempfile
import unittest
import shutil
from pathlib import Path

import numpy as np

from polypbench.analysis.candidate_stage import (
    ReturnObserver, NativeCapture, mask_statistics, stage_metrics, to_original,
    valid_cell_fraction, summarize_image,
    aggregate_reports,
)
from polypbench.analysis.supervision_eligibility import box_support, region_eligibility, deterministic_seed


class StageGeometryTests(unittest.TestCase):
    def test_padding_scale_is_native_unclipped_geometry(self):
        box = np.array([[.5, .5, 1, 1]], dtype=np.float32)
        actual = to_original(box, (832, 1024), (800, 1000), (400, 500))
        np.testing.assert_allclose(actual, [[0, 0, 512, 416]])

    def test_marginal_coverage_is_not_exclusive_matching(self):
        gt = np.array([[0, 0, 10, 10], [0, 0, 10, 10]])
        matrix, report = stage_metrics(np.array([[0, 0, 10, 10], [np.nan, 0, 1, 1]]), gt, [27, 28])
        self.assertEqual(report["thresholds"]["0.75"]["one_to_one_coverage"], 1)
        self.assertEqual(report["thresholds"]["0.75"]["marginal_gt_covered"], 2)
        self.assertEqual(report["best_query_or_token_id"], [27, 27])
        self.assertEqual(report["invalid_or_degenerate_boxes"], 1)
        self.assertTrue(np.isfinite(matrix).all())

    def test_empty_pools_and_negative_images(self):
        matrix, report = stage_metrics(np.empty((0, 4)), np.array([[0, 0, 1, 1]]))
        self.assertEqual(report["max_iou"], [0])
        self.assertEqual(report["best_query_or_token_id"], [None])
        self.assertEqual(matrix.shape, (1, 0))
        _, negative = stage_metrics(np.array([[0, 0, 1, 1]]), np.empty((0, 4)))
        self.assertEqual(negative["max_iou"], [])

    def test_mask_area_records_native_padding_foreground(self):
        binary = np.array([[[1, 1], [0, 0]], [[0, 0], [0, 0]]], dtype=bool)
        fraction = valid_cell_fraction((2, 2), (8, 8), (8, 4))
        stats = mask_statistics(binary, fraction)
        np.testing.assert_equal(stats["foreground_pixels"], [2, 0])
        np.testing.assert_equal(stats["padding_foreground_fraction"], [.5, 0])
        np.testing.assert_equal(stats["empty"], [False, True])
        np.testing.assert_equal(stats["one_row"], [True, False])

    def test_profiler_observes_returns_without_changing_rng_or_outputs(self):
        import torch
        def calculation(x):
            intermediate = x * 2
            return intermediate + 1
        records = []
        state = torch.random.get_rng_state().clone()
        def callback(loc, result):
            records.append((loc["intermediate"].clone(), result.clone()))
        with ReturnObserver({calculation.__code__: callback}):
            actual = calculation(torch.ones(3))
        torch.testing.assert_close(actual, torch.full((3,), 3.))
        torch.testing.assert_close(records[0][0], torch.full((3,), 2.))
        self.assertIsNone(sys.getprofile())
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        with self.assertRaises(ValueError):
            with ReturnObserver({}):
                raise ValueError("test")
        self.assertIsNone(sys.getprofile())

    def test_semantic_unique_box_support_does_not_assign_instance_identity(self):
        mask = np.ones((4, 4), dtype=bool)
        support = box_support(mask, [[0, 0, 3, 4], [1, 0, 4, 4]])
        self.assertEqual(support[0]["semantic_pixels_unique_box"], 4)
        self.assertEqual(support[0]["semantic_pixels_ambiguous_boxes"], 8)
        fg = np.zeros((64, 64), bool); fg[16:48, 16:48] = True
        ambiguous = region_eligibility(fg, np.ones_like(fg), False)
        self.assertGreater(ambiguous["core_pixels"], 0)
        self.assertFalse(ambiguous["pixel_eligible"])
        self.assertFalse(any(g["eligible"] for g in ambiguous["grids"].values()))
        self.assertNotEqual(deterministic_seed(42, 1), deterministic_seed(42, 2))


class NativeDecoderCPUTests(unittest.TestCase):
    """Real native forward/TopK/mask2box/pred_box; stub only attention layers.

    Exercises the observer's private-local assumptions without any GPU forward.
    It does not claim validation of attention or remote checkpoint parity.
    """
    def test_native_stage_mapping_and_empty_mask_sentinel(self):
        import torch
        from torch import nn
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "third_party/MaskDINO"))
        from maskdino.modeling.transformer_decoder.maskdino_decoder import MaskDINODecoder
        from maskdino.utils.box_ops import masks_to_boxes
        torch.set_num_threads(1)
        torch.random.default_generator.manual_seed(19)
        decoder = MaskDINODecoder.__new__(MaskDINODecoder)
        nn.Module.__init__(decoder)
        decoder.num_feature_levels = 4
        decoder.hidden_dim = 8
        decoder.num_queries = 300
        decoder.num_layers = 9
        decoder.two_stage = True
        decoder.initial_pred = True
        decoder.initialize_box_type = "mask2box"
        decoder.learn_tgt = False
        decoder.dn = "no"
        decoder.mask_classification = True
        decoder.input_proj = nn.ModuleList([nn.Identity() for _ in range(4)])
        decoder.enc_output = nn.Linear(8, 8)
        decoder.enc_output_norm = nn.LayerNorm(8)
        decoder.decoder_norm = nn.LayerNorm(8)
        decoder.class_embed = nn.Linear(8, 1)
        decoder.mask_embed = nn.Linear(8, 4)
        decoder._bbox_embed = nn.Linear(8, 4)
        decoder.bbox_embed = nn.ModuleList([nn.Linear(8, 4) for _ in range(9)])
        class AttentionStub(nn.Module):
            def forward(self, tgt, refpoints_unsigmoid, **unused):
                return ([tgt.transpose(0, 1)] * 9,
                        [refpoints_unsigmoid.sigmoid().transpose(0, 1)] * 10)
        decoder.decoder = AttentionStub()
        decoder.eval()
        x = [torch.randn(1, 8, 32, 32), torch.randn(1, 8, 16, 16), torch.randn(1, 8, 8, 8), torch.randn(1,8,4,4)]
        mask_features = torch.randn(1, 4, 32, 32)
        from maskdino.maskdino import MaskDINO
        class BackboneStub(nn.Module):
            def forward(self, images):
                return x
        class HeadStub(nn.Module):
            def __init__(self):
                super().__init__()
                self.predictor = decoder
                self.num_classes = 1
            def forward(self, features):
                return self.predictor(features, mask_features, None)
        model = MaskDINO.__new__(MaskDINO)
        nn.Module.__init__(model)
        model.backbone = BackboneStub()
        model.sem_seg_head = HeadStub()
        model.size_divisibility = 32
        model.register_buffer("pixel_mean", torch.zeros(3,1,1))
        model.register_buffer("pixel_std", torch.ones(3,1,1))
        model.num_queries = 300
        model.sem_seg_postprocess_before_inference = True
        model.semantic_on = model.panoptic_on = model.focus_on_box = False
        model.instance_on = True
        model.test_topk_per_image = 100
        model.eval()
        capture = NativeCapture(model)
        inputs = [{"image": torch.zeros(3,120,128), "height": 120, "width": 128}]
        with torch.no_grad():
            plain = model(inputs)[0]["instances"]
            rng = torch.random.get_rng_state().clone()
            with ReturnObserver(capture.handlers):
                observed = model(inputs)[0]["instances"]
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for key in ("pred_boxes", "scores", "pred_masks"):
            a, b = plain.get(key), observed.get(key)
            torch.testing.assert_close(a.tensor if key == "pred_boxes" else a,
                                       b.tensor if key == "pred_boxes" else b, rtol=0, atol=0)
        self.assertEqual(capture.data["dense_boxes"].shape, (1360, 4))
        # Current native empty-mask output is a sentinel, not a zero box.
        sentinel = masks_to_boxes(torch.zeros(1, 4, 4, dtype=torch.bool))
        np.testing.assert_equal(sentinel.numpy(), [[1e8, 1e8, 0, 0]])
        data = capture.data
        self.assertEqual(data["padded_hw"], (128,128))
        self.assertEqual(data["mapped_hw"], (120,128))
        np.testing.assert_allclose(data["native_top100_class_scores"] * data["native_top100_mask_scores"], data["native_top100_scores"], rtol=1e-6)
        image = {"id": 1, "height": 120, "width": 128, "file_name": "fake.jpg", "center": "C1", "sampling_stratum": "test"}
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            saved = temp / "source"; saved.mkdir()
            report = summarize_image(data, image, [{"id": 1, "bbox": [20,20,40,40]}], saved)
            self.assertIn("decoder_09->final_top100", report["transitions"])
            arrays = np.load(saved / "1.npz")
            self.assertEqual(arrays["decoder_init_gt_iou"].shape, (1,300))
            from polypbench.io import write_json
            from polypbench.analysis.candidate_stage_report import report as make_report
            from PIL import Image
            Image.new("RGB", (128,120), "white").save(temp / "fake.jpg")
            for label in ("baseline", "mssl"):
                target = temp / label / "images"
                shutil.copytree(saved, target)
                write_json(target.parent / "verification.json", {"status": "passed"})
                aggregate_reports([report], target.parent)
            make_report(temp, temp, temp / "comparison")
            self.assertTrue((temp / "comparison/comparison.json").is_file())


if __name__ == "__main__":
    unittest.main()
