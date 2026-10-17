"""Real R50 model/optimizer on CPU. Forward tests reduce depth/queries, disable DN.

Deformable attention uses upstream's existing CPU reference fallback, not a test
double. These tests do not claim CUDA AMP or CUDA-only denoising validation.
"""
from dataclasses import replace
from pathlib import Path
import copy
import gc
import sys
import unittest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"third_party/MaskDINO"))

from methods.scda_v3.config import build_cfg
from methods.scda_v3.experiment import load_experiment
from methods.scda_v3.integration import register

CPU_ALIGNMENT_EVIDENCE = {}


def config(enabled=True, small=False):
    cfg = build_cfg(load_experiment(ROOT/"methods/scda_v3/configs/v3a.yaml"), "/tmp/v3-cpu-unused", "cpu").clone()
    cfg.defrost()
    if not enabled:
        cfg.MODEL.META_ARCHITECTURE = "MaskDINO"
        cfg.MODEL.MaskDINO.TRANSFORMER_DECODER_NAME = "MaskDINODecoder"
        cfg.MODEL.SCDA_V3.ENABLED = False
    if small:
        cfg.MODEL.MaskDINO.DN = "no"
        cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES = 8
        cfg.MODEL.MaskDINO.DEC_LAYERS = 2
        cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 1
        cfg.MODEL.MaskDINO.TRAIN_NUM_POINTS = 32
        cfg.TEST.DETECTIONS_PER_IMAGE = 8
    cfg.freeze()
    return cfg


def batch():
    from detectron2.structures import Instances, Boxes
    torch.manual_seed(12)
    instance = Instances((64, 64))
    instance.gt_boxes = Boxes(torch.tensor([[8., 8., 56., 56.]]))
    instance.gt_classes = torch.zeros(1, dtype=torch.long)
    instance.gt_masks = torch.zeros(1, 64, 64, dtype=torch.bool)
    instance.gt_masks[:, 10:54, 10:54] = True
    instance.gt_scda_reliable = torch.ones(1, dtype=torch.bool)
    instance.gt_scda_ids = torch.ones(1, dtype=torch.long)
    return [{"image": torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8), "instances": instance,
             "padding_mask": torch.zeros(64, 64, dtype=torch.bool), "scda_original_gt_count": 1}]


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        register(ROOT/"third_party/MaskDINO")

    def test_canonical_model_initialization_rng_parameter_and_optimizer_groups(self):
        import train_net
        from detectron2.modeling import build_model
        torch.manual_seed(42); native = build_model(config(False))
        reference = {name: tensor.clone() for name, tensor in native.state_dict().items()}
        rng = torch.get_rng_state().clone()
        native_count = sum(p.numel() for p in native.parameters())
        del native; gc.collect()
        cfg = config()
        torch.manual_seed(42); model = build_model(cfg)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(native_count, 52017477)
        for name, tensor in model.state_dict().items():
            if not name.startswith("aux_box_head."):
                self.assertTrue(torch.equal(reference[name], tensor), name)
        native_head = model.sem_seg_head.predictor._bbox_embed
        for a, b in zip(native_head.parameters(), model.aux_box_head.parameters()):
            self.assertIsNot(a, b); self.assertNotEqual(a.data_ptr(), b.data_ptr())
            self.assertTrue(torch.equal(a, b))
        optimizer = train_net.Trainer.build_optimizer(cfg, model)
        actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(actual), len(set(actual)))
        self.assertEqual(set(actual), {id(p) for p in model.parameters() if p.requires_grad})
        aux_ids = {id(p) for p in model.aux_box_head.parameters()}
        for group in optimizer.param_groups:
            if any(id(p) in aux_ids for p in group["params"]): self.assertEqual(group["lr"], .0001)
        self.assertEqual(sum(p.numel() for p in model.aux_box_head.parameters()), 132612)
        self.assertFalse(torch.cuda.is_initialized())

    def test_real_cpu_native_loss_gradient_eval_alignment_and_auxiliary_gradient_routes(self):
        from detectron2.modeling import build_model
        from detectron2.utils.events import EventStorage
        from methods.scda_v3.diagnostics import record_audit
        from unittest.mock import patch
        torch.manual_seed(42); native = build_model(config(False, True))
        torch.manual_seed(42); model = build_model(config(True, True))
        model.v3_options = replace(model.v3_options, lambda_aux=0., diagnostics=False)
        inputs = batch()
        native.train(); model.train()
        torch.manual_seed(19); na = native(inputs); sum(na.values()).backward()
        reference_grads = {name: p.grad.clone() for name, p in native.named_parameters() if p.grad is not None}
        # The upstream CPU grid_sample backward can differ even in native-native
        # repeats. Check a repeat against the SAME fixed norm budget as V3.
        native.zero_grad(set_to_none=True)
        torch.manual_seed(19); repeat = native(inputs); sum(repeat.values()).backward()
        maxima = {"native_repeat_relative_l2": 0., "v3_relative_l2": 0.}
        def compare_gradient(name, reference, actual, label):
            difference = float((reference.double() - actual.double()).norm())
            norm = float(reference.double().norm())
            self.assertLessEqual(difference, 1e-6 + 1e-5 * norm, name)
            maxima[label] = max(maxima[label], difference / norm if norm else difference)
        for name, p in native.named_parameters():
            if name in reference_grads:
                compare_gradient(name, reference_grads[name], p.grad, "native_repeat_relative_l2")
        torch.manual_seed(19); vb = model(inputs); sum(vb.values()).backward()
        self.assertEqual(set(na), set(vb)-{"loss_v3_box"})
        for key in na: torch.testing.assert_close(na[key], vb[key], rtol=0, atol=0)
        native_grads = dict(native.named_parameters())
        for name, p in model.named_parameters():
            if name.startswith("aux_box_head."):
                self.assertIsNotNone(p.grad); self.assertEqual(float(p.grad.abs().sum()), 0)
            elif native_grads[name].grad is None:
                self.assertIsNone(p.grad, name)
            else:
                compare_gradient(name, reference_grads[name], p.grad, "v3_relative_l2")
        CPU_ALIGNMENT_EVIDENCE.update(maxima, gradient_rtol=1e-5, gradient_atol=1e-6,
                                      loss_tolerance="exact", scope="reduced CPU network, DN off")
        native.eval(); model.eval()
        with torch.no_grad():
            a, b = native(inputs)[0]["instances"], model(inputs)[0]["instances"]
        for key, value in a.get_fields().items():
            other = b.get_fields()[key]
            torch.testing.assert_close(value.tensor if hasattr(value, "tensor") else value,
                                       other.tensor if hasattr(other, "tensor") else other, rtol=0, atol=0)
        model.train(); model.zero_grad(set_to_none=True)
        model.v3_options = replace(model.v3_options, lambda_aux=.05)
        # Native final box layer is zero-initialized. Its initial memory gradient
        # is mathematically zero; exercise the later nonzero-weight regime on
        # this disposable test model without an optimizer/training step.
        with torch.no_grad(): model.aux_box_head.layers[-1].weight.fill_(.001)
        evidence = {}
        def inspect(current, losses, auxiliary, payload, targets, assignments):
            self.assertGreater(assignments[0].reg.token_indices.numel(), 0)
            params = list(current.named_parameters())
            grads = torch.autograd.grad(auxiliary["loss_v3_box"], [p for _, p in params],
                                        allow_unused=True, retain_graph=True)
            for prefix in ("aux_box_head.", "backbone.", "sem_seg_head.pixel_decoder."):
                evidence[prefix] = sum(float(g.abs().sum()) for (name, p), g in zip(params, grads)
                                       if name.startswith(prefix) and g is not None)
                self.assertGreater(evidence[prefix], 0, prefix)
            for (name, p), g in zip(params, grads):
                if name.startswith("sem_seg_head.predictor.") and not name.startswith(
                        ("sem_seg_head.predictor.enc_output.", "sem_seg_head.predictor.enc_output_norm.")):
                    self.assertTrue(g is None or not g.any(), name)
            memory_grad = torch.autograd.grad(auxiliary["loss_v3_box"], payload["output_memory"], retain_graph=True)[0]
            self.assertGreater(float(memory_grad.abs().sum()), 0)
        model.v3_audit_callback = inspect
        for policy in ("legacy_quality_iou", "mask_multiscale"):
            model.v3_options = replace(model.v3_options, assignment_policy=policy)
            model.zero_grad(set_to_none=True)
            result = model(inputs)
            self.assertTrue(all(torch.isfinite(v) for v in result.values()))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__": unittest.main()
