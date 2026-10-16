"""Aggregate measured assignments; no research GO thresholds or label changes."""
import math


def validate_candidate_availability(summary, options):
    """Inactive objectives and non-nested quality selections are intentional."""
    counts = summary["nonempty_valid_reliable_gt"]
    for task, weight in (("cls", options.lambda_cls), ("reg", options.lambda_box)):
        if weight > 0 and not counts[f"{task}_positive_count"]:
            raise AssertionError(f"No candidates for active auxiliary {task}")
    if (options.reg_selection in ("coverage_topk", "core_plus_transition") and
            counts["cls_subset_reg_violations"]):
        raise AssertionError("Nested-set policy violated")
    return {"passed": True, "active_cls": options.lambda_cls > 0,
            "active_box": options.lambda_box > 0, "reg_selection": options.reg_selection}


def validate_head_gradients(report, options):
    """Zero lambda must give zero weighted gradients, not a false failure/GO."""
    for head, objective, weight in (("class_embed", "cls_weighted", options.lambda_cls),
                                    ("bbox_embed", "box_weighted", options.lambda_box)):
        values = report["parameter_space"][head]["gradient_norms"]
        if any(not math.isfinite(values[name]) for name in ("cls_weighted", "box_weighted")):
            raise FloatingPointError("Non-finite auxiliary gradient")
        if weight > 0 and values[objective] <= 0:
            raise AssertionError(f"No active auxiliary gradient to shared {head}")
        if weight == 0 and values[objective] != 0:
            raise AssertionError(f"Inactive auxiliary {objective} has a nonzero gradient")
    return {"passed": True}


def validate_v2_assignments(images):
    """Exact confirmed quota identities, not empirical performance thresholds."""
    checked = 0
    for image in images:
        if image.get("reg_selection") != "core_plus_transition":
            raise AssertionError("Expected explicit F-v2 audit policy")
        for entry in image["per_gt"]:
            if "reg_shortfall" not in entry:  # unreliable/empty/invalid GT skipped
                if entry["n_cls_selected"] or entry["n_reg_selected"]:
                    raise AssertionError("Skipped GT received supervision")
                continue
            cls, reg = entry["cls_token_indices"], entry["reg_token_indices"]
            expected_cls = min(2, entry["n_cls_owned"])
            expected_medium = min(2, entry["n_transition_owned"])
            if (len(cls) != expected_cls or len(reg) != expected_cls + expected_medium or
                    reg[:len(cls)] != cls or len(set(reg)) != len(reg)):
                raise AssertionError("F-v2 quota/reuse/no-refill identity violated")
            if (any(value < .75 for value in entry["cls_coverage"]) or
                    any(not .25 <= value < .75 for value in entry["reg_coverage"][len(cls):]) or
                    entry["reg_coverage"][:len(cls)] != entry["cls_coverage"]):
                raise AssertionError("F-v2 coverage band identity violated")
            checked += 1
    return {"passed": True, "checked_gt_count": checked,
            "rule": "actual cls core (<=2) + medium (<=2); no cross-band refill"}


def summarize_decoupling(images):
    entries = [entry for image in images for entry in image["per_gt"]]
    reliable = [entry for entry in entries if entry["reliable"]]
    eligible = [entry for entry in reliable if entry["valid_box"] and entry["nonempty_mask"]]

    def group(items):
        cls_count = sum(entry["n_cls_selected"] for entry in items)
        reg_count = sum(entry["n_reg_selected"] for entry in items)
        transition = sum(entry.get("reg_transition_count", 0) for entry in items)
        reg_only = sum(len(set(entry.get("reg_token_indices", [])) -
                           set(entry.get("cls_token_indices", []))) for entry in items)
        return {
            "gt_count": len(items), "cls_positive_count": cls_count, "reg_positive_count": reg_count,
            "cls_empty_gt_count": sum(entry["n_cls_selected"] == 0 for entry in items),
            "reg_empty_gt_count": sum(entry["n_reg_selected"] == 0 for entry in items),
            "both_empty_gt_count": sum(entry["n_cls_selected"] == entry["n_reg_selected"] == 0 for entry in items),
            "both_empty_gt_fraction": sum(entry["n_cls_selected"] == entry["n_reg_selected"] == 0 for entry in items) / len(items) if items else None,
            "reg_transition_fraction": transition / reg_count if reg_count else None,
            "reg_only_fraction": reg_only / reg_count if reg_count else None,
            "threshold_changes_gt_count": sum(entry.get("reg_threshold_changes_selection", False) for entry in items),
            "cls_subset_reg_violations": sum(not entry.get("cls_subset_reg", True) for entry in items),
            "quota_evaluated_gt_count": sum("reg_shortfall" in entry for entry in items),
            "reg_shortfall_gt_count": sum(entry.get("reg_shortfall", 0) > 0 for entry in items),
            "reg_core_shortfall_gt_count": sum((entry.get("reg_core_shortfall") or 0) > 0 for entry in items),
            "reg_transition_shortfall_gt_count": sum((entry.get("reg_transition_shortfall") or 0) > 0 for entry in items),
            "reg_core_selected_count": sum(entry.get("n_reg_core_selected", 0) for entry in items),
            "reg_transition_selected_count": transition,
        }

    return {"images": len(images), "all_augmented_gt": group(entries),
            "source_reliable_gt": group(reliable), "nonempty_valid_reliable_gt": group(eligible),
            "original_gt_count": sum(image.get("n_original_gt") or 0 for image in images),
            "original_multibox_images": sum((image.get("n_original_gt") or 0) > 1 for image in images),
            "original_multibox_image_fraction": sum((image.get("n_original_gt") or 0) > 1 for image in images) / len(images) if images else None,
            "reg_selection": sorted({image.get("reg_selection", "coverage_topk") for image in images}),
            "note": "Overlap=1 is expected for nested selection. F-v2 reserves two regression slots for medium coverage without cross-band refill. Counterfactual retains the active policy, not a top4-core replacement.",
            "per_image": images}
