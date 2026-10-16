"""Policy-aware acceptance guards, with no model or GPU dependencies."""
from types import SimpleNamespace
import unittest

from methods.scda_proposal.audit_summary import validate_candidate_availability, validate_head_gradients


class PolicyPrecheckTests(unittest.TestCase):
    def options(self, policy="quality_iou", cls_weight=0.):
        return SimpleNamespace(reg_selection=policy, lambda_cls=cls_weight, lambda_box=.05)

    def summary(self, reg_count=4, violations=1):
        return {"nonempty_valid_reliable_gt": {"cls_positive_count": 0,
                "reg_positive_count": reg_count, "cls_subset_reg_violations": violations}}

    def test_quality_reg_only_does_not_require_active_cls_or_nested_sets(self):
        validate_candidate_availability(self.summary(), self.options())
        validate_candidate_availability(self.summary(), self.options("quality_iou_boxgate"))
        with self.assertRaises(AssertionError):
            validate_candidate_availability(self.summary(reg_count=0), self.options())

    def test_active_cls_and_legacy_nested_contracts_are_not_relaxed(self):
        with self.assertRaises(AssertionError):
            validate_candidate_availability(self.summary(), self.options(cls_weight=.05))
        with self.assertRaises(AssertionError):
            validate_candidate_availability(self.summary(), self.options("core_plus_transition"))

    def test_zero_lambda_requires_exact_zero_but_active_box_requires_gradient(self):
        def report(cls=0., box=1.):
            return {"parameter_space": {name: {"gradient_norms": {
                    "cls_weighted": cls, "box_weighted": box}}
                    for name in ("class_embed", "bbox_embed")}}
        validate_head_gradients(report(), self.options())
        for invalid in (report(cls=.1), report(box=0.), report(box=float("nan"))):
            with self.assertRaises((AssertionError, FloatingPointError)):
                validate_head_gradients(invalid, self.options())


if __name__ == "__main__":
    unittest.main()
