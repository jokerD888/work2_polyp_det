import tempfile
import unittest
import sys
from types import SimpleNamespace
from pathlib import Path
from polypbench.io import read_json
from polypbench.analysis.initialization_intervention import signed_events, comparison


class InterventionAccountingTests(unittest.TestCase):
    def test_native_infinite_logits_are_finite_sigmoid_references(self):
        import torch
        from polypbench.analysis.initialization_intervention import InitializationIntervention
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"third_party/MaskDINO"))
        decoder = SimpleNamespace(training=False, two_stage=True, initialize_box_type="mask2box", initial_pred=True, learn_tgt=False)
        intervention = InitializationIntervention(decoder)
        encoder = torch.zeros(1,300,4); encoder[0,0,0] = float("inf")
        mask = torch.zeros_like(encoder)
        arguments = (encoder, mask, torch.arange(300)[None], torch.zeros(1,300,8), torch.zeros(1,300,2,2))
        self.assertIs(intervention.replace(*arguments), mask)
        intervention.condition = "encoder"
        self.assertIs(intervention.replace(*arguments), encoder)
        intervention.condition = "blend05"
        self.assertTrue(torch.isfinite(intervention.replace(*arguments)).all())
        encoder[0,0,0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "normalized reference"):
            intervention.replace(*arguments)

    def test_recovery_is_not_net_benefit(self):
        self.assertEqual(signed_events([.1,.8,.9,.4], [.7,.2,.3,.8], .5),
            {"recovered": 2, "newly_lost": 2, "net": 0})
        self.assertEqual(signed_events([], [], .9), {"recovered": 0, "newly_lost": 0, "net": 0})

    def test_final_losses_are_not_hidden_by_initialization_recovery(self):
        def image(values):
            return {"image_id": 1, "annotation_ids": [11,12], "gt": 2, "center": "C5",
                "gt_metadata": [{"relative_area_quartile": "q1"}, {"relative_area_quartile": "q4"}],
                "stages": {s: {"max_iou": v} for s,v in zip(
                    ("encoder_top300","decoder_init","decoder_09","final_top100"), values)}}
        native = image([[.8,.8],[.2,.9],[.2,.9],[.2,.9]])
        other = image([[.8,.8],[.7,.8],[.7,.3],[.7,.3]])
        with tempfile.TemporaryDirectory() as temporary:
            result = comparison({"native": [native], "encoder": [other], "blend05": [other]}, Path(temporary))
            events = result["encoder"]["all"]["stages"]["decoder_09"]["0.5"]
            self.assertEqual(events, {"recovered": 1, "newly_lost": 1, "net": 0})
            self.assertEqual(result["encoder"]["all"]["historical_init_lost"]["alternative_final300_covered"], 1)
            rows = read_json(Path(temporary)/"per_gt_comparison.json")
            self.assertEqual(len(rows), 4)


if __name__ == "__main__":
    unittest.main()
