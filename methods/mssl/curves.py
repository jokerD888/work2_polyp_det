"""Pre-training scalar curves on fixed source-training GT, never target data."""
import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from polypbench.io import read_json, write_json
from .config import load_experiment, MSSLOptions
from .loss import side_losses


def run(config, output):
    torch.set_num_threads(2)
    cfg = load_experiment(config)
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    data = read_json(cfg["dataset"]["train"]["annotations"])
    grouped = {}
    for ann in data["annotations"]:
        grouped.setdefault(ann["image_id"], []).append(ann)
    selected = [im for im in sorted(data["images"], key=lambda x: x["id"])
                if len(grouped.get(im["id"], [])) == 1][:3]
    options = MSSLOptions(**cfg["mssl"])
    errors = np.linspace(-1, 1, 401)
    report = {"source": "first three original single-box training images by image_id",
              "normalization": "d=(predicted edge-GT edge)/max(GT axis size,4px)",
              "curve_scope": "individual side before four-side/eligible averaging and lambda; raw source masks, no augmentation",
              "samples": []}
    for image in selected:
        ann = grouped[image["id"]][0]
        x, y, w, h = ann["bbox"]
        target = torch.tensor([x, y, x+w, y+h], dtype=torch.float32)
        with Image.open(Path(cfg["dataset"]["raw_mask_root"]) / Path(image["file_name"]).name) as im:
            mask = torch.from_numpy((np.asarray(im.convert("L")) > 127).copy())
        scale = torch.tensor([max(w,4), max(h,4)]*2)
        variants = {}
        for label, support, form in (("Mask-CDF", "mask", "cdf"), ("Box-CDF", "box", "cdf"),
                                     ("Mask-static", "mask", "static")):
            opt = replace(options, support=support, loss_form=form)
            values, grads = [], []
            for d in errors:
                pred = (target + float(d)*scale).requires_grad_()
                losses = side_losses(pred, target, mask, opt)
                grad = torch.autograd.grad(losses.sum(), pred)[0] * scale
                values.append(losses.detach().tolist()); grads.append(grad.tolist())
            variants[label] = {"loss": values, "gradient_wrt_d": grads}
        fig, axes = plt.subplots(2,4, figsize=(14,6), sharex=True)
        for side, name in enumerate(("left", "top", "right", "bottom")):
            for label, values in variants.items():
                axes[0,side].plot(errors, np.array(values["loss"])[:,side], label=label)
                axes[1,side].plot(errors, np.array(values["gradient_wrt_d"])[:,side], label=label)
            axes[0,side].set_title(name)
            axes[1,side].set_xlabel("normalized signed edge error d")
            for ax in axes[:,side]: ax.axvline(0,color="grey",lw=.5); ax.grid(alpha=.2)
        axes[0,0].set_ylabel("side loss"); axes[1,0].set_ylabel("dL/dd")
        axes[0,0].legend(fontsize=8)
        fig.suptitle(f"Source train image {image['id']} - fixed GT; no tuning")
        fig.tight_layout(); fig.savefig(out/f"image_{image['id']}.png", dpi=150)
        fig.savefig(out/f"image_{image['id']}.pdf"); plt.close(fig)
        assert all(np.isfinite(v[k]).all() for v in variants.values() for k in ("loss","gradient_wrt_d"))
        report["samples"].append({"image_id": image["id"], "bbox": ann["bbox"],
                                  "errors": errors.tolist(), "variants": variants})
    report["status"] = "passed"
    write_json(out/"curves.json", report)
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--output",required=True)
    args=parser.parse_args(); run(args.config,args.output)
