import unittest
from polypbench.analysis.candidate_stage_local import dependency_decision


class ContinuationGateTests(unittest.TestCase):
    def setUp(self):
        self.order = ["E", "F", "external_evaluation", "comparison"]
        self.state = {"pipeline_pid": 10, "status": "running", "completed": ["E"]}
        self.queue = {"status": "running", "tasks": [
            {"name": name, "status": "completed" if name == "E" else "pending"} for name in self.order]}

    def decision(self, actual_start="start"):
        return dependency_decision(self.state, self.queue, self.order, 10, "start", actual_start)

    def complete(self):
        self.state.update(status="completed", completed=self.order.copy())
        self.queue.update(status="completed")
        for task in self.queue["tasks"]:
            task["status"] = "completed"

    def test_running_predecessor_never_releases_gpu(self):
        self.assertEqual(self.decision(), "waiting_predecessor")
        with self.assertRaises(RuntimeError):
            self.decision(None)

    def test_all_tasks_and_controller_exit_are_required(self):
        self.complete()
        self.assertEqual(self.decision(), "waiting_predecessor_exit")
        self.assertEqual(self.decision(None), "ready")
        self.queue["tasks"][-1]["status"] = "pending"
        with self.assertRaises(RuntimeError):
            self.decision(None)

    def test_failure_pause_and_reused_pid_cannot_launch(self):
        for status in ("failed", "paused", "stopped"):
            self.state["status"] = status
            with self.assertRaises(RuntimeError):
                self.decision()
        self.state["status"] = "running"
        with self.assertRaises(RuntimeError):
            self.decision("unrelated_start")

    def test_final_state_queue_write_gap_is_safe(self):
        self.complete(); self.queue["status"] = "running"
        self.assertEqual(self.decision(), "waiting_completion_write")
        with self.assertRaises(RuntimeError):
            self.decision(None)


if __name__ == "__main__":
    unittest.main()
