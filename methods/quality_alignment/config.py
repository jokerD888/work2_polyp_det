OPTIONS = {
    "enabled": True,
    "variant": "IA-BCE-Final",
    "quality_alpha": .25,
    "negative_gamma": 2.,
    "target_min": .01,
    "target_detached": True,
    "scope": "final_normal_decoder_classification_only",
}


def validate(experiment):
    options = experiment.get("quality_alignment")
    if not isinstance(options, dict) or set(options) != set(OPTIONS):
        raise ValueError("Exact pinned quality_alignment options are required")
    if type(options["enabled"]) is not bool:
        raise ValueError("enabled must be bool")
    for key, value in OPTIONS.items():
        if key != "enabled" and options[key] != value:
            raise ValueError(f"Unregistered IA-BCE variant: {key}")
    if "box_regression" in experiment or experiment.get("gradient_probe_period", 0):
        raise ValueError("No CIoU combination or overlapping probe in this experiment")
    return experiment
