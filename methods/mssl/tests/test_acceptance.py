import json
from pathlib import Path
import tempfile
import unittest
from methods.mssl.acceptance import smoke_gate


class AcceptanceTests(unittest.TestCase):
    def test_evaluation_only_last_record_is_not_training_record(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for stage in ("train","predict_val"):
                (root/stage).mkdir()
                (root/stage/"state.json").write_text(json.dumps({"status":"completed"}))
            (root/"predict_val/metrics.json").write_text('{}')
            rows=[{"iteration":15,"total_loss":7.,"loss_mssl":.01,
                   "optimizer/successful_updates":6},
                  {"iteration":15,"bbox/AP":0.01}]
            (root/"train/metrics.json").write_text('\n'.join(json.dumps(r) for r in rows))
            gate=smoke_gate(root)
            self.assertEqual(gate["successful_updates"],6)
            self.assertEqual(gate["status"],"passed")
