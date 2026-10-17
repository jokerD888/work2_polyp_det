"""Instrumented training entry for collapse-cause diagnosis.

Reuses methods.scda_proposal.worker unchanged; patches observation-only
hooks onto the shared trainer bases (see diag.py module docstring). The
standard pipeline (protocol, data, losses, optimizer, schedule) is identical;
diagnostics write to side sinks under <output>/diag/.
"""
import argparse
from pathlib import Path
import os
import sys

HOLDER = {"experiment": None, "level": "full", "force": None}
DIAG = None


def _patch_trainer_bases():
    from detectron2.engine import AMPTrainer
    from detectron2.engine.train_loop import SimpleTrainer
    from methods.scda_proposal.diag import CollapseDiagnosis

    original_run_step = AMPTrainer.__dict__["run_step"]
    original_write_metrics = SimpleTrainer.__dict__["_write_metrics"]

    def run_step(self):
        if DIAG is not None:
            DIAG.pre_step()
        original_run_step(self)
        if DIAG is not None:
            DIAG.post_step(self.grad_scaler)
            DIAG.periodic()

    def _write_metrics(self, loss_dict, data_time):
        if DIAG is not None:
            DIAG.capture_losses(loss_dict)
        original_write_metrics(self, loss_dict, data_time)

    AMPTrainer.run_step = run_step
    AMPTrainer._write_metrics = _write_metrics

    from detectron2.engine import DefaultTrainer as D2DefaultTrainer

    original_build_model = D2DefaultTrainer.build_model.__func__

    def build_model(cls, config):
        global DIAG
        model = original_build_model(cls, config)
        if DIAG is None and HOLDER["level"] is not None:
            DIAG = CollapseDiagnosis(Path(config.OUTPUT_DIR) / "diag", model,
                                     HOLDER["experiment"], HOLDER["level"], HOLDER["force"])
            if DIAG.level == "full":
                DIAG.install_model_hooks()
        return model

    D2DefaultTrainer.build_model = classmethod(build_model)

    # build_optimizer is defined on train_net.Trainer (MaskDINO's own AdamW +
    # full-model clipping wrapper); the worker's local subclass super() hits it.
    import train_net

    original_build_optimizer = train_net.Trainer.build_optimizer.__func__

    def build_optimizer(cls, config, model):
        optimizer = original_build_optimizer(cls, config, model)
        if DIAG is not None:
            import types
            original_step = optimizer.step

            def step(opt_self, closure=None):
                DIAG.note_step_called()
                return original_step(closure)

            optimizer.step = types.MethodType(step, optimizer)
        return optimizer

    train_net.Trainer.build_optimizer = classmethod(build_optimizer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("train", "predict"), required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--checkpoint")
    parser.add_argument("--diag-level", choices=("none", "full"), default="full")
    parser.add_argument("--force-trigger-iter", type=int, default=None)
    parser.add_argument("--worker-module", default="methods.scda_proposal.worker")
    args = parser.parse_args()
    if args.mode != "train":
        raise SystemExit("diag_worker only trains; use the standard entry for predict")
    HOLDER["experiment"] = str(Path(args.experiment).resolve())
    HOLDER["level"] = args.diag_level
    HOLDER["force"] = args.force_trigger_iter

    experiment = __import__("json").loads(Path(args.experiment).read_text())
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    sys.path.insert(0, experiment["repo"]["path"])
    import importlib
    W = importlib.import_module(args.worker_module)

    _patch_trainer_bases()
    from detectron2.engine import launch
    launch(W.worker_main, 1, dist_url="auto", args=(args,))


if __name__ == "__main__":
    main()
