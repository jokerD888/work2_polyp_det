"""H2-A: fixed-weight gradient competition diagnostics, no optimizer step.

For the F-Reg-Quality LR=1e-4 model at several checkpoints (init / best4674 /
ckpt5224 / final5499 — TRUE mid-training checkpoints were not saved by the
keep-2 policy; this is declared, not substituted), over a fixed set of 24
training batches (bs16, seed 42), compute with torch.autograd.grad on one
shared forward:
  g_S : loss_scda_box  (lambda_box=0.05 applied; F-Reg lambda_cls=0)
  g_E : weight_dict["loss_bbox_interm"] * loss_bbox_interm
  g_D : sum of weighted decoder box losses (final + aux layers, DN excluded)
  g_N : full weighted native loss (cls/mask/dice/DN/interm/aux)
and report norms, ratio ||g_S||/||g_N|| and cosine similarities, globally and
per module bucket (shared _bbox_embed / enc_output / input_proj / pixel
decoder / backbone / transformer decoder / ...).

An fp16-autocast cross-check (one batch at the best checkpoint) probes how far
the fp32 diagnostic is from the AMP training regime; actual AdamW composite
updates are separately observable in the run's own diag records.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from harness_diag import (DECODER_BOX_KEYS, DN_BOX_KEYS, FREG_5224, FREG_BEST,
                          FREG_FINAL, ROOT, build_experiment, build_model,
                          json_dump, load_weights, make_batches,
                          map_fixed_records, module_of, register_dataset, seed_all)

OUT = Path(__file__).parent


def grad_of(scalar, params):
    grads = torch.autograd.grad(scalar, params, retain_graph=True, allow_unused=True)
    return [torch.zeros_like(p) if g is None else g for p, g in zip(params, grads)]


def gvec_stats(g):
    g = torch.cat([x.reshape(-1) for x in g])
    return g, float(g.float().norm())


def cos(g1, g2):
    n = float(g1.float().norm()) * float(g2.float().norm())
    return float((g1 @ g2).float()) / n if n > 0 else None


def bucket_norms(named_params, grads):
    buckets = {}
    for (name, _), g in zip(named_params, grads):
        b = module_of(name)
        buckets.setdefault(b, 0.0)
        buckets[b] += float(g.float().norm() ** 2)
    return {k: v ** 0.5 for k, v in buckets.items()}


def cos_in_bucket(named_params, g1, g2, bucket):
    v1 = torch.cat([g.reshape(-1) for (name, _), g in zip(named_params, g1)
                    if module_of(name) == bucket])
    v2 = torch.cat([g.reshape(-1) for (name, _), g in zip(named_params, g2)
                    if module_of(name) == bucket])
    return cos(v1, v2)


def run_batch(model, chunk, options):
    from methods.scda_proposal.scda_assigner import assign_scda
    from methods.scda_proposal.scda_loss import scda_losses
    from harness_diag import to_model_inputs, scda_targets
    images, gt_instances = to_model_inputs(model, chunk)
    targets = model.prepare_targets(gt_instances, images)
    features = model.backbone(images.tensor)
    outputs, mask_dict = model.sem_seg_head(features, targets=targets)
    payload = outputs.pop("scda_payload")
    losses = model.criterion(outputs, targets, mask_dict)
    aux_targets = scda_targets(model, chunk, gt_instances, images, targets)
    assignments = assign_scda(payload, aux_targets, options)
    aux = scda_losses(payload, aux_targets, assignments, options,
                      model.scda_box_weight, model.scda_giou_weight)
    wd = model.criterion.weight_dict
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]

    L_S = aux.losses["loss_scda_box"]
    L_E = wd["loss_bbox_interm"] * losses["loss_bbox_interm"]
    L_D = sum(wd[k] * losses[k] for k in DECODER_BOX_KEYS if k in losses)
    L_D_dn = L_D + sum(wd[k] * losses[k] for k in DN_BOX_KEYS if k in losses)
    L_N = sum(v * losses[k] for k, v in wd.items() if k in losses)

    gS, gE, gD, gDdn, gN = (grad_of(L, params) for L in (L_S, L_E, L_D, L_D_dn, L_N))

    flat = {k: gvec_stats(g) for k, g in
            [("S", gS), ("E", gE), ("D", gD), ("Ddn", gDdn), ("N", gN)]}
    rec = {
        "losses": {"S": float(L_S), "E": float(L_E), "D": float(L_D),
                   "D_dn": float(L_D_dn), "N": float(L_N)},
        "norms": {k: v[1] for k, v in flat.items()},
        "ratio_S_over_N": flat["S"][1] / max(flat["N"][1], 1e-12),
        "cos_S_N": cos(flat["S"][0], flat["N"][0]),
        "cos_S_E": cos(flat["S"][0], flat["E"][0]),
        "cos_S_D": cos(flat["S"][0], flat["D"][0]),
        "cos_S_Ddn": cos(flat["S"][0], flat["Ddn"][0]),
        "cos_E_D": cos(flat["E"][0], flat["D"][0]),
        "cos_D_Ddn": cos(flat["D"][0], flat["Ddn"][0]),
        "norms_by_bucket": {k: bucket_norms(named, g) for k, g in
                            [("S", gS), ("E", gE), ("D", gD), ("N", gN)]},
    }
    rec["cos_S_N_by_bucket"] = {b: cos_in_bucket(named, gS, gN, b)
                                for b in sorted({module_of(n) for n, _ in named})}
    rec["cos_S_D_by_bucket"] = {b: cos_in_bucket(named, gS, gD, b)
                                for b in sorted({module_of(n) for n, _ in named})}
    return rec


def aggregate(records):
    def mean(key):
        vals = [r[key] for r in records if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None
    out = {"n_batches": len(records),
           "ratio_S_over_N_mean": mean("ratio_S_over_N"),
           "cos_S_N_mean": mean("cos_S_N"), "cos_S_E_mean": mean("cos_S_E"),
           "cos_S_D_mean": mean("cos_S_D"), "cos_S_Ddn_mean": mean("cos_S_Ddn"),
           "cos_E_D_mean": mean("cos_E_D")}
    for key in ("S", "E", "D", "N"):
        vals = [r["norms"][key] for r in records]
        out[f"norm_{key}_mean"] = sum(vals) / len(vals)
        out[f"norm_{key}_max"] = max(vals)
    buckets = sorted(records[0]["norms_by_bucket"]["N"].keys())
    for b in buckets:
        out[f"bucket_{b}"] = {
            "norm_N_mean": sum(r["norms_by_bucket"]["N"][b] for r in records) / len(records),
            "norm_S_mean": sum(r["norms_by_bucket"]["S"][b] for r in records) / len(records),
            "cos_S_N_mean": (lambda vs: sum(v for v in vs if v is not None) / max(1, len([v for v in vs if v is not None])))([r["cos_S_N_by_bucket"][b] for r in records]),
            "cos_S_D_mean": (lambda vs: sum(v for v in vs if v is not None) / max(1, len([v for v in vs if v is not None])))([r["cos_S_D_by_bucket"][b] for r in records]),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit-images", type=int, default=0, help="debug: map only N images")
    args = parser.parse_args()

    seed_all(42)
    model, cfg, experiment = build_model()
    from methods.scda_proposal.config import SCDAOptions
    options = SCDAOptions()
    dicts = register_dataset(experiment)
    if args.limit_images:
        dicts = dicts[:args.limit_images]
    print(f"mapping {len(dicts)} train images...", flush=True)
    records = map_fixed_records(cfg, experiment, dicts)
    batches = make_batches(records, args.batch_size, args.batches, seed=42)
    print(f"{len(batches)} fixed batches x {args.batch_size}", flush=True)

    checkpoints = [("init", None), ("best4674", FREG_BEST),
                   ("ckpt5224", FREG_5224), ("final5499", FREG_FINAL)]
    results = {"note_mid_checkpoints": "not saved by the keep-2 policy; "
               "best4674/ckpt5224/final5499 are the only trained states available",
               "batch_size": args.batch_size, "n_batches": len(batches), "checkpoints": {}}
    for label, ckpt in checkpoints:
        info = load_weights(model, ckpt)
        model.train()
        recs = []
        for bi, idx in enumerate(batches):
            chunk = [records[i] for i in idx]
            model.zero_grad(set_to_none=False)
            rec = run_batch(model, chunk, options)
            recs.append(rec)
            if bi % 8 == 0:
                print(f"  [{label}] batch {bi}/{len(batches)} "
                      f"cos_S_N={rec['cos_S_N']:.4f} r={rec['ratio_S_over_N']:.4f}", flush=True)
        agg = aggregate(recs)
        results["checkpoints"][label] = {"loaded": info, "aggregate": agg, "per_batch": recs}
        print(f"=== {label}: r_S/N={agg['ratio_S_over_N_mean']:.4f} "
              f"cos(S,N)={agg['cos_S_N_mean']:.4f} cos(S,E)={agg['cos_S_E_mean']:.4f} "
              f"cos(S,D)={agg['cos_S_D_mean']:.4f}", flush=True)

    # AMP cross-check at best4674, one batch
    load_weights(model, FREG_BEST)
    model.train()
    idx = batches[0]
    chunk = [records[i] for i in idx]
    with torch.autocast("cuda", dtype=torch.float16):
        rec_amp = run_batch(model, chunk, options)
    results["amp_crosscheck_best4674_batch0"] = rec_amp
    json_dump(OUT / "h2a_results.json", results)
    print("H2A_DONE")


if __name__ == "__main__":
    main()
