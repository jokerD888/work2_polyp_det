import copy
import unittest
from pathlib import Path

from scripts.recover_support_reference_queue import derive_protocol


class RecoveryPlanTests(unittest.TestCase):
    def setUp(self):
        self.protocol = {'order': ['geometry_formal_gate', 'support_train', 'support_source_predict'],
            'runs': {'blend05': 'a', 'geometry': 'b', 'support': 'c'},
            'configs': {'support': 'frozen.yaml'}, 'source_sha256': {'model.py': 'same'}}
        self.state = {'stage': 'support_train', 'completed': ['geometry_formal_gate']}

    def test_only_incomplete_run_changes_and_partial_not_selected(self):
        original = copy.deepcopy(self.protocol)
        p = derive_protocol(self.protocol, self.state, Path('/old'), 'stamp')
        self.assertEqual(self.protocol, original)
        self.assertEqual(p['runs'], {'blend05': 'a', 'geometry': 'b', 'support': 'c-restart-stamp'})
        self.assertEqual(p['configs'], original['configs'])
        self.assertEqual(p['source_sha256'], original['source_sha256'])
        self.assertTrue(p['recovery']['no_partial_checkpoint_selection'])

    def test_wrong_stage_cannot_silently_skip_work(self):
        self.state['stage'] = 'geometry_train'
        with self.assertRaises(ValueError):
            derive_protocol(self.protocol, self.state, Path('/old'), 'stamp')

    def test_missing_completed_control_rejected(self):
        self.state['completed'] = []
        with self.assertRaises(ValueError):
            derive_protocol(self.protocol, self.state, Path('/old'), 'stamp')


if __name__ == '__main__':
    unittest.main()
