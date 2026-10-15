"""CPU-only, no-model replay of real LSJ images before/after zero mask padding."""
import argparse
import random
import sys
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="methods/scda_proposal/configs/kvasir_v2.yaml")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="runs/lsj-instance-padding-cpu-audit.json")
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Preserve the previous audit: {output}")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from methods.scda_proposal.experiment import load_experiment
    from polypbench.io import sha256, write_json
    experiment = load_experiment(args.config)
    sys.path.insert(0, experiment["repo"]["path"])

    import numpy as np
    import torch
    from detectron2.data import detection_utils as utils, transforms as T
    from detectron2.data.datasets import load_coco_json
    from maskdino.data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import build_transform_gen
    from polypbench.adapters.bitmask_mapper import BitmaskLSJMapper
    from methods.scda_proposal.mapper import SCDALSJMapper

    # Only the mapper settings are needed; no model or CUDA context is created.
    settings = experiment["overrides"]
    cfg = SimpleNamespace(INPUT=SimpleNamespace(
        IMAGE_SIZE=settings["INPUT.IMAGE_SIZE"], MIN_SCALE=settings["INPUT.MIN_SCALE"],
        MAX_SCALE=settings["INPUT.MAX_SCALE"], RANDOM_FLIP=settings["INPUT.RANDOM_FLIP"],
        FORMAT="RGB"))
    entry = experiment["dataset"]["train"]
    records = load_coco_json(entry["annotations"], entry["image_root"], extra_annotation_keys=["id"])
    baseline = BitmaskLSJMapper(cfg, "annotations")
    scda = SCDALSJMapper(cfg, "single_instance")
    legacy = BitmaskLSJMapper(cfg, "annotations")
    legacy.augmentations = build_transform_gen(cfg, True)
    oracle_augmentations = build_transform_gen(cfg, True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    order = torch.randperm(len(records), generator=torch.Generator().manual_seed(args.seed)).tolist()[:args.samples]
    rows = []
    for index in order:
        record = records[index]
        state = (random.getstate(), np.random.get_state(), torch.get_rng_state())

        def replay(call):
            random.setstate(state[0])
            np.random.set_state(state[1])
            torch.set_rng_state(state[2])
            return call()

        old = replay(lambda: legacy(record))
        new = replay(lambda: baseline(record))
        modified = replay(lambda: scda(record))

        def geometry_reference():
            image = utils.read_image(record["file_name"], format=cfg.INPUT.FORMAT)
            ones = np.ones(image.shape[:2], dtype=np.uint8)
            image, transforms = T.apply_transform_gens(oracle_augmentations, image)
            # Independently distinguish label 1 from native padding label 255.
            # Unlike bool(), this does not mistake semantic ignore for valid data.
            return transforms.apply_segmentation(ones) != 1

        padding = torch.from_numpy(replay(geometry_reference))
        torch.testing.assert_close(new["padding_mask"], padding, rtol=0, atol=0)
        for key in ("image", "padding_mask"):
            torch.testing.assert_close(new[key], modified[key], rtol=0, atol=0)
        torch.testing.assert_close(new["image"], old["image"], rtol=0, atol=0)
        for other in (old, modified):
            torch.testing.assert_close(new["instances"].gt_boxes.tensor,
                                       other["instances"].gt_boxes.tensor, rtol=0, atol=0)
            torch.testing.assert_close(new["instances"].gt_classes,
                                       other["instances"].gt_classes, rtol=0, atol=0)
        clean = new["instances"].gt_masks
        previous = old["instances"].gt_masks
        torch.testing.assert_close(clean, modified["instances"].gt_masks, rtol=0, atol=0)
        torch.testing.assert_close(clean[:, ~padding], previous[:, ~padding], rtol=0, atol=0)
        assert not clean[:, padding].any(), "Padding became foreground"
        rows.append({"image_id": int(record["image_id"]), "instances": len(clean),
                     "padding_pixels": int(padding.sum()),
                     "legacy_reported_padding_pixels": int(old["padding_mask"].sum()),
                     "legacy_padding_foreground_instance_pixels": int(previous[:, padding].sum()),
                     "legacy_foreground_instance_pixels": int(previous.sum()),
                     "fixed_foreground_instance_pixels": int(clean.sum())})

    summary = {"images": len(rows), "images_with_padding": sum(r["padding_pixels"] > 0 for r in rows),
               "legacy_images_reporting_padding": sum(r["legacy_reported_padding_pixels"] > 0 for r in rows),
               "padding_pixels": sum(r["padding_pixels"] for r in rows),
               "legacy_padding_foreground_instance_pixels": sum(r["legacy_padding_foreground_instance_pixels"] for r in rows),
               "legacy_foreground_instance_pixels": sum(r["legacy_foreground_instance_pixels"] for r in rows),
               "fixed_foreground_instance_pixels": sum(r["fixed_foreground_instance_pixels"] for r in rows)}
    result = {"status": "passed", "device": "cpu", "model_forward": False,
              "seed": args.seed, "dataset_sha256": sha256(entry["annotations"]),
              "mapper_settings": vars(cfg.INPUT), "torch": torch.__version__,
              "source_sha256": {path: sha256(root / path) for path in (
                  "polypbench/adapters/instance_lsj.py", "polypbench/adapters/bitmask_mapper.py",
                  "methods/scda_proposal/mapper.py", "scripts/check_lsj_instance_padding.py")},
              "verified": ["independent padding geometry", "baseline/SCDA exact equality",
                           "legacy/fixed image and supplied box equality", "nonpad masks unchanged",
                           "all fixed padding is background"],
              "summary": summary,
              "examples": sorted(rows, key=lambda r: r["legacy_padding_foreground_instance_pixels"], reverse=True)[:5]}
    write_json(output, result)
    print(summary)
    print(f"CPU audit passed: {output}")


if __name__ == "__main__":
    main()
