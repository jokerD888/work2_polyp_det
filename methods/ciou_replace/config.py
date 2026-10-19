"""One fixed replacement variant, shared original training protocol."""
OPTIONS = {"kind": "ciou_replace", "epsilon": 1e-7,
           "alpha_stop_gradient": True, "iou_gate": False,
           "scope": "all_native_box_regression", "compute_dtype": "float32"}


def validate(experiment):
    if experiment.get("box_regression") != OPTIONS:
        raise ValueError("CIoU-Replace options must equal the prespecified contract")
    if experiment["mssl"]["enabled"] or experiment.get("gradient_probe_period", 0):
        raise ValueError("CIoU must not enable MSSL or its training gradient probe")
    return experiment
