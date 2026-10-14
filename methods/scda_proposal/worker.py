"""Dedicated model-environment worker; importing it launches nothing."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from polypbench.io import read_json, sha256, write_json


def worker_main(args):
    import torch
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
    from detectron2.data.datasets import load_coco_json
    from detectron2.engine import default_setup, hooks
    from detectron2.evaluation import COCOEvaluator
    from detectron2.projects.deeplab import add_deeplab_config
    from detectron2.utils import comm
    from detectron2.utils.events import get_event_storage
    from detectron2.utils.logger import setup_logger

    experiment = read_json(args.experiment)
    sys.path.insert(0, experiment["repo"]["path"])
    import train_net as native
    from .config import add_scda_config
    from .integration import configure_model, register
    register(experiment["repo"]["path"])
    for split in ("train", "val", "test"):
        if split not in experiment["dataset"]:
            continue
        entry = experiment["dataset"][split]
        name = f"bench_{split}"
        annotations, image_root = entry["annotations"], entry["image_root"]
        DatasetCatalog.register(name, lambda a=annotations, r=image_root, n=name: load_coco_json(
            a, r, n, extra_annotation_keys=["id", "scda_reliable"]))
        MetadataCatalog.get(name).set(json_file=annotations, image_root=image_root, evaluator_type="coco")
    cfg = get_cfg()
    add_deeplab_config(cfg)
    native.add_maskdino_config(cfg)
    add_scda_config(cfg)
    cfg.merge_from_file(experiment["model"]["config"])
    for key, value in experiment["overrides"].items():
        cfg.merge_from_list([key, repr(value)])
    for key, value in experiment["scda"].items():
        setattr(cfg.MODEL.SCDA, key.upper(), value)
    options = configure_model(cfg)
    cfg.DATASETS.TRAIN = ("bench_train",)
    cfg.DATASETS.TEST = (f"bench_{'val' if args.mode == 'train' else args.split}",)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = experiment["model"]["num_classes"]
    cfg.MODEL.WEIGHTS = args.checkpoint if args.mode == "predict" else experiment["model"]["weights"]
    cfg.OUTPUT_DIR = args.output
    cfg.SEED = experiment["seed"]
    cfg.TEST.AUG.ENABLED = False
    cfg.TEST.DETECTIONS_PER_IMAGE = experiment["model"].get("detections_per_image", 100)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="maskdino")
    if comm.is_main_process():
        write_json(Path(args.output) / "environment.json", {
            "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "packages": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()})
        write_json(Path(args.output) / "method.json", {
            "method": "F-v1 MIPS-SCDA", "options": experiment["scda"],
            "box_weight": cfg.MODEL.MaskDINO.BOX_WEIGHT,
            "giou_weight": cfg.MODEL.MaskDINO.GIOU_WEIGHT,
            "box_coefficients": "reuse resolved native coefficients once, then lambda_box",
            "baseline_config_sha256": experiment["baseline_config_sha256"],
            "precheck_status": "unverified; this file is not a GO result"})
        if args.mode == "train" and cfg.MODEL.WEIGHTS:
            from detectron2.utils.file_io import PathManager
            cached = PathManager.get_local_path(cfg.MODEL.WEIGHTS)
            write_json(Path(args.output) / "initialization.json", {"source": cfg.MODEL.WEIGHTS, "sha256": sha256(cached)})

    class Trainer(native.Trainer):
        @classmethod
        def build_model(cls, config):
            model = super().build_model(config)
            if comm.is_main_process():
                from .diagnostics import parameter_audit
                write_json(Path(args.output) / "model_parameters.json", parameter_audit(model))
            return model

        @classmethod
        def build_optimizer(cls, config, model):
            optimizer = super().build_optimizer(config, model)
            if comm.is_main_process():
                from .diagnostics import parameter_audit
                write_json(Path(args.output) / "optimizer_parameters.json", parameter_audit(model, optimizer))
            return optimizer

        @classmethod
        def build_train_loader(cls, config):
            if options.enabled:
                from .mapper import SCDALSJMapper
                mapper = SCDALSJMapper(config, options.mask_policy)
            else:
                from polypbench.adapters.bitmask_mapper import BitmaskLSJMapper
                mapper = BitmaskLSJMapper(config, experiment["dataset"]["training_boxes"])
            return build_detection_train_loader(config, mapper=mapper)

        @classmethod
        def build_evaluator(cls, config, dataset_name, output_folder=None):
            return COCOEvaluator(dataset_name, tasks=("bbox",), distributed=True,
                                 output_dir=output_folder or str(Path(config.OUTPUT_DIR) / "inference"),
                                 use_fast_impl=False)

        def run_step(self):
            if options.diagnostics:
                get_event_storage().scda_output_dir = self.cfg.OUTPUT_DIR
            return super().run_step()

        def build_hooks(self):
            existing = super().build_hooks()
            if comm.is_main_process() and "checkpoint_keep" in experiment:
                for index, hook in enumerate(existing):
                    if isinstance(hook, hooks.PeriodicCheckpointer):
                        existing[index] = hooks.PeriodicCheckpointer(
                            self.checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD,
                            max_to_keep=experiment["checkpoint_keep"])
            if comm.is_main_process() and experiment.get("checkpoint_selection") == "best_val_ap":
                class RecordedBestCheckpointer(hooks.BestCheckpointer):
                    def _best_checking(inner):
                        previous = (inner.best_metric, inner.best_iter)
                        super()._best_checking()
                        if (inner.best_metric, inner.best_iter) != previous:
                            write_json(Path(args.output) / "best_checkpoint.json", {
                                "file": "model_best.pth", "split": "val", "metric": "bbox/AP",
                                "metric_percent": inner.best_metric, "iteration": inner.best_iter,
                                "tie_break": "earliest", "rule": "maximum validation AP50:95"})
                best = RecordedBestCheckpointer(self.cfg.TEST.EVAL_PERIOD, self.checkpointer,
                                                 "bbox/AP", mode="max", file_prefix="model_best")
                for index, hook in enumerate(existing):
                    if isinstance(hook, hooks.EvalHook):
                        existing.insert(index + 1, best)
                        break
                else:
                    raise RuntimeError("Best-checkpoint selection requires a validation EvalHook")
            return existing

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
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("train", "predict"), required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    cfg = read_json(args.experiment)
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    sys.path.insert(0, cfg["repo"]["path"])
    from detectron2.engine import launch
    launch(worker_main, cfg["runtime"]["num_gpus"], dist_url="auto", args=(args,))


if __name__ == "__main__":
    main()
