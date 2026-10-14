"""Aggregate measured assignments; no research GO thresholds or label changes."""


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
        }

    return {"images": len(images), "all_augmented_gt": group(entries),
            "source_reliable_gt": group(reliable), "nonempty_valid_reliable_gt": group(eligible),
            "original_gt_count": sum(image.get("n_original_gt") or 0 for image in images),
            "original_multibox_images": sum((image.get("n_original_gt") or 0) > 1 for image in images),
            "original_multibox_image_fraction": sum((image.get("n_original_gt") or 0) > 1 for image in images) / len(images) if images else None,
            "note": "Overlap=1 is expected for nonempty cls with the confirmed nested selection. No new thresholds or candidate rules applied.",
            "per_image": images}
