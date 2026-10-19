"""Executed in the detector's isolated Python environment, including each DDP rank."""
import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from polypbench.io import read_json, sha256, write_json


def native_import(cfg):
    sys.path.insert(0, cfg["repo"]["path"])
    import train_net
    return train_net


def doctor(cfg):
    import torch
    native_import(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in model environment")
    if torch.cuda.device_count() < cfg["runtime"]["num_gpus"]:
        raise RuntimeError("Fewer visible GPUs than runtime.num_gpus")
    # Exercise both forward/backward, not just import the compiled extension.
    from maskdino.modeling.pixel_decoder.ops.functions.ms_deform_attn_func import MSDeformAttnFunction, ms_deform_attn_core_pytorch
    torch.manual_seed(17)
    value = torch.rand(1, 4, 1, 4, device="cuda", requires_grad=True)
    shapes = torch.tensor([[2, 2]], dtype=torch.long, device="cuda")
    starts = torch.tensor([0], dtype=torch.long, device="cuda")
    locations = torch.rand(1, 2, 1, 1, 2, 2, device="cuda", requires_grad=True)
    weights = torch.full((1, 2, 1, 1, 2), .5, device="cuda", requires_grad=True)
    actual = MSDeformAttnFunction.apply(value, shapes, starts, locations, weights, 1)
    expected = ms_deform_attn_core_pytorch(value, shapes, locations, weights)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    inputs = (value, locations, weights)
    actual_grad = torch.autograd.grad(actual, inputs, torch.ones_like(actual), retain_graph=True)
    expected_grad = torch.autograd.grad(expected, inputs, torch.ones_like(expected))
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
    return {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "compute_capability": torch.cuda.get_device_capability(), "detectron2": importlib.metadata.version("detectron2"), "cuda_extension_forward_backward": "passed"}


def worker_main(args):
    import torch
    from detectron2.data.datasets import register_coco_instances
    from detectron2.engine import default_setup
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from detectron2.evaluation import COCOEvaluator
    from detectron2.utils import comm
    from detectron2.utils.logger import setup_logger
    from detectron2.checkpoint import DetectionCheckpointer

    experiment = read_json(args.experiment)
    native = native_import(experiment)
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from .config import build_cfg
    from .data import load_records
    for split in ("train", "val", "test"):
        if split in experiment["dataset"]:
            entry = experiment["dataset"][split]
            name = f"bench_{split}"
            if split == "train":
                DatasetCatalog.register(name, lambda e=entry,n=name: load_records(
                    e["annotations"], e["image_root"], n, experiment["dataset"]["raw_mask_root"]))
                MetadataCatalog.get(name).set(json_file=entry["annotations"], image_root=entry["image_root"], evaluator_type="coco")
            else:
                register_coco_instances(name, {}, entry["annotations"], entry["image_root"])
    cfg = build_cfg(experiment, args.output).clone()
    cfg.defrost()
    cfg.DATASETS.TEST = (f"bench_{'val' if args.mode == 'train' else args.split}",)
    cfg.MODEL.WEIGHTS = args.checkpoint if args.mode == "predict" else experiment["model"]["weights"]
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="maskdino")
    if comm.is_main_process():
        write_json(Path(args.output) / "environment.json", {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "packages": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()})
        if args.mode == "train" and cfg.MODEL.WEIGHTS:
            from detectron2.utils.file_io import PathManager
            cached = PathManager.get_local_path(cfg.MODEL.WEIGHTS)
            write_json(Path(args.output) / "initialization.json", {"source": cfg.MODEL.WEIGHTS, "sha256": sha256(cached)})

    class Trainer(native.Trainer):
        def build_hooks(self):
            from detectron2.engine import hooks
            existing = super().build_hooks()
            if comm.is_main_process() and "checkpoint_keep" in experiment:
                for index, hook in enumerate(existing):
                    if isinstance(hook, hooks.PeriodicCheckpointer):
                        existing[index] = hooks.PeriodicCheckpointer(self.checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD, max_to_keep=experiment["checkpoint_keep"])
            if comm.is_main_process() and experiment.get("checkpoint_selection") == "best_val_ap":
                class RecordedBestCheckpointer(hooks.BestCheckpointer):
                    def _best_checking(inner):
                        previous = (inner.best_metric, inner.best_iter)
                        super()._best_checking()
                        if (inner.best_metric, inner.best_iter) != previous:
                            write_json(Path(args.output) / "best_checkpoint.json", {"file": "model_best.pth", "split": "val", "metric": "bbox/AP", "metric_percent": inner.best_metric, "iteration": inner.best_iter, "tie_break": "earliest", "rule": "maximum validation AP50:95"})
                best = RecordedBestCheckpointer(self.cfg.TEST.EVAL_PERIOD, self.checkpointer, "bbox/AP", mode="max", file_prefix="model_best")
                for index, hook in enumerate(existing):
                    if isinstance(hook, hooks.EvalHook):
                        existing.insert(index + 1, best)
                        break
                else:
                    raise RuntimeError("Best-checkpoint selection requires a validation EvalHook")
            return existing

        def __init__(self, config):
            super().__init__(config)
            from .accumulation import AccumulatingAMPTrainer
            steps = config.SOLVER.IMS_PER_BATCH // experiment["runtime"]["micro_batch"]
            probe = None
            if experiment.get("gradient_probe_period", 0):
                from .gradient_probe import GradientProbe
                probe = GradientProbe(self.model, Path(args.output) / "gradient_probes.jsonl",
                                      experiment["gradient_probe_period"])
            self._trainer = AccumulatingAMPTrainer(self.model, self.data_loader, self.optimizer, steps, probe)

        @classmethod
        def build_train_loader(cls, config):
            from detectron2.data import build_detection_train_loader
            from .data import MSSLMapper
            micro = config.clone()
            micro.defrost()
            micro.SOLVER.IMS_PER_BATCH = experiment["runtime"]["micro_batch"]
            micro.freeze()
            return build_detection_train_loader(micro, mapper=MSSLMapper(micro, "annotations"))

        @classmethod
        def build_evaluator(cls, config, dataset_name, output_folder=None):
            # Keep native inference and ID remapping, export predictions, evaluate bbox only.
            return COCOEvaluator(dataset_name, tasks=("bbox",), distributed=True, output_dir=output_folder or str(Path(config.OUTPUT_DIR) / "inference"), use_fast_impl=False)

    if comm.is_main_process():
        write_json(Path(args.output) / "method.json", {
            "method": "CIoU-Replace" if "box_regression" in experiment else "MSSL V1",
            "options": experiment.get("box_regression", experiment["mssl"]),
            "effective_batch": cfg.SOLVER.IMS_PER_BATCH,
            "micro_batch": experiment["runtime"]["micro_batch"],
            "accumulation_steps": cfg.SOLVER.IMS_PER_BATCH // experiment["runtime"]["micro_batch"],
            "normalization": "native per microbatch, including DN; same method/control; not exact physical-batch equivalence",
            "mask_policy": "raw semantic for original single-box images in BOTH groups; multi-box native unchanged, no extra loss",
            "final_decoder_only": "box_regression" not in experiment,
            "regression_log_key_semantics": "CIoU replaces loss_giou*; matching remains GIoU" if "box_regression" in experiment else "native GIoU",
            "new_learned_parameters": 0})
    if args.mode == "train":
        trainer = Trainer(cfg)
        trainer.resume_or_load(resume=False)
        return trainer.train()
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    return Trainer.test(cfg, model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--mode", choices=["train", "predict", "doctor"], required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    cfg = read_json(args.experiment)
    # PyTorch 2.6 changed torch.load's default. Native Detectron2/fvcore checkpoints
    # contain numpy/optimizer state; use the legacy loader within this worker only.
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    if args.mode == "doctor":
        print(json.dumps(doctor(cfg), indent=2))
        return
    sys.path.insert(0, cfg["repo"]["path"])
    from detectron2.engine import launch
    # Auto-selected local port; registration happens inside each launched rank.
    launch(worker_main, cfg["runtime"]["num_gpus"], dist_url="auto", args=(args,))


if __name__ == "__main__":
    main()
