"""Observation-only collapse diagnostics for instrumented training runs.

Adds no parameters, consumes no RNG, and does not alter any loss value,
gradient, update, or data order. All records go to side sinks under the
run's train/diag/ directory; the standard metrics.json/console pipeline is
untouched. While instrumented, diagnostics.record_training_audit (a
metrics-only path) is neutralized; training math is unchanged.

Levels:
  none  passive: per-iter file_names + RNG fingerprints only.
  full  passive + raw per-iter losses, focal decomposition, actual
        num_boxes, gradient norms (global/per-module, pre/post clip),
        update norm, AMP scaler scale and skip detection, selection/
        top-Q/matching switching, trigger controller with snapshots and
        independent-process snapshot evaluation.
"""
import hashlib
import json
import math
import random
import subprocess
import sys
import zlib
from collections import deque
from pathlib import Path

import numpy as np
import torch


class JsonlSink:
    def __init__(self, path):
        self.path = Path(path)
        self._f = None

    def write(self, record):
        if self._f is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._f = self.path.open("a")
        self._f.write(json.dumps(record, default=float, allow_nan=True) + "\n")
        self._f.flush()


def _crc(t):
    return zlib.crc32(t.numpy().tobytes())


class CollapseDiagnosis:
    """Attach to one training run. Observation-only; see module docstring."""

    ALERT_CE = 1.0          # lightweight log alert (no action)
    TRIGGER_CE = 2.0        # sustained raw-CE trigger
    TRIGGER_CE_ITERS = 3
    GRAD_SPIKE_FACTOR = 5.0
    COOLDOWN = 200
    EVAL_BUDGET = 8
    SNAP_SCHEDULE = (3, 10)
    MODULE_PREFIXES = (("backbone", "backbone"), ("pixel_decoder", "encoder"), ("predictor", "decoder"))

    def __init__(self, out_dir, model, experiment_json, level="full", force_trigger_iter=None):
        assert level in ("none", "full")
        self.level = level
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.sink = JsonlSink(self.dir / "diag_metrics.jsonl")
        self.sel_sink = JsonlSink(self.dir / "diag_selections.jsonl")
        self.events = JsonlSink(self.dir / "diag_events.jsonl")
        self.snap_dir = self.dir / "snapshots"
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.experiment_json = str(experiment_json)
        self.force_trigger_iter = force_trigger_iter
        self.iter = -1
        self.file_names = []
        self.rng_fp = {}
        self.current_losses = {}
        self.current_gn = {}
        self.current_num_boxes = None
        self.ce_stats = {}
        self.current_scale = 1.0
        self.step_called = False
        self.prev_params = None
        self.prev_state = None
        self.grad_history = deque(maxlen=200)
        self.ce_high_run = 0
        self.cool_until = -1
        self.trigger_count = 0
        self.eval_count = 0
        self.pending_snaps = set()
        self.last_visit = {}
        self._install_forward_capture()
        self.events.write({"iter": -1, "kind": "setup", "level": level,
                           "force_trigger_iter": force_trigger_iter})

    # ---------- passive (both levels) ----------
    def _install_forward_capture(self):
        model, diag = self.model, self

        original_forward = model.forward

        def forward(batched_inputs):
            if not model.training:
                # evaluation forwards (in-train hooks) are not training iters
                return original_forward(batched_inputs)
            diag.iter += 1
            diag.file_names = [r.get("file_name", "?") for r in batched_inputs]
            diag.rng_fp = {"rng_torch": _crc(torch.random.get_rng_state()),
                           "rng_cuda": _crc(torch.cuda.get_rng_state()),
                           "rng_np": _crc(torch.as_tensor(np.asarray(np.random.get_state()[1]))),
                           "rng_py": zlib.crc32(repr(random.getstate()).encode())}
            if diag.level == "none":
                diag.sink.write({"iter": diag.iter, "kind": "passive",
                                 "file_names": diag.file_names,
                                 **diag.rng_fp})
            return original_forward(batched_inputs)

        model.forward = forward

    # ---------- full level ----------
    def install_model_hooks(self):
        import methods.scda_proposal.diagnostics as diag_mod
        diag_mod.record_training_audit = lambda *a, **k: None
        # Native MaskDINO (SCDA disabled, e.g. DCNv4-only) has no SCDA model
        # attributes; the selection-audit callback only exists on the SCDA
        # meta-architecture. Clip/criterion/update hooks below are agnostic.
        if getattr(self.model, "scda_options", None) is not None:
            self.model.scda_options = dataclasses_replace(self.model.scda_options, diagnostics=True)
            self.model.scda_audit_callback = self._audit_callback
        self._wrap_clip_hook()
        self._wrap_criterion()

    def _audit_callback(self, current, losses, auxiliary, payload, targets, assignments):
        self.current_num_boxes = sum(int(t.labels.numel()) for t in targets)
        if self.level != "full":
            return
        rec = {"iter": self.iter, "num_boxes": self.current_num_boxes}
        for b, a in enumerate(assignments):
            fn = self.file_names[b] if b < len(self.file_names) else str(b)
            tokens = tuple(sorted(a.reg.token_indices.tolist()))
            topq = tuple(sorted(payload["topk_indices"][b].tolist()))
            prev = self.last_visit.get(fn)
            switch = None if prev is None else len(set(tokens) ^ set(prev[0])) / max(1, len(tokens))
            self.last_visit[fn] = (tokens, topq)
            rec[fn] = {"sel_hash": hashlib.sha1(repr(tokens).encode()).hexdigest()[:12],
                       "topq_hash": hashlib.sha1(repr(topq).encode()).hexdigest()[:12],
                       "switch_frac": switch}
        self.sel_sink.write(rec)

    def _wrap_clip_hook(self):
        import torch.nn.utils as U
        diag = self
        id2mod = {}
        for full_name, p in self.model.named_parameters():
            mod = "other"
            for prefix, name in self.MODULE_PREFIXES:
                if full_name.startswith(prefix):
                    mod = name
                    break
            id2mod[id(p)] = mod
        original = U.clip_grad_norm_

        def clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None):
            # Materialize ONCE and pass the SAME list to the original: consuming
            # the caller's generator would leave it empty and silently disable
            # clipping (incident 2026-10-17, caught by global_pre==0 with
            # non-empty per-module norms).
            params = [p for p in parameters if p.grad is not None]
            sq = {}
            nonfinite = False
            for p in params:
                mod = id2mod.get(id(p), "other")
                n = float(p.grad.float().norm())
                sq[mod] = sq.get(mod, 0.0) + n ** 2
                nonfinite = nonfinite or not math.isfinite(n)
            total = original(params, max_norm, norm_type)
            diag.current_gn = {"global_pre": float(total),
                               "post": min(float(total), float(max_norm)),
                               "nonfinite": nonfinite or not bool(torch.isfinite(total)),
                               **{f"gn_{k}": v ** 0.5 for k, v in sq.items()}}
            return total

        U.clip_grad_norm_ = clip_grad_norm_

    def _wrap_criterion(self):
        criterion = self.model.criterion
        diag = self
        original = criterion.loss_labels

        def loss_labels(outputs, targets, indices, num_boxes, log=True):
            loss = original(outputs, targets, indices, num_boxes, log)
            with torch.no_grad():
                p = outputs["pred_logits"].detach().float().sigmoid()[..., 0]
                matched = torch.zeros_like(p, dtype=torch.bool)
                src_hashes = []
                for b, (src_idx, _) in enumerate(indices):
                    matched[b, torch.as_tensor(src_idx, device=p.device, dtype=torch.long)] = True
                    src_hashes.append(hash(tuple(src_idx)))
                neg_p = p[~matched]
                alpha, gamma = float(criterion.focal_alpha), 2.0
                # Record only the first call per forward (final-layer main CE);
                # DN and per-layer aux calls reuse loss_labels afterwards.
                if not getattr(diag, "_ce_first_done", False):
                    diag._ce_first_done = True
                    diag.ce_stats = {
                        "n_queries": int(p.shape[1]),
                        "num_boxes": int(num_boxes),
                        "n_matched": int(matched.sum()),
                        "pos_focal_sum": float((alpha * (1 - p[matched]) ** gamma *
                                                (-torch.log(p[matched].clamp(min=1e-6)))).sum()),
                        "neg_focal_sum": float(((1 - alpha) * neg_p ** gamma *
                                                (-torch.log((1 - neg_p).clamp(min=1e-6)))).sum()) if neg_p.numel() else 0.0,
                        "unmatched_p_p50": float(neg_p.median()) if neg_p.numel() else None,
                        "unmatched_p_max": float(neg_p.max()) if neg_p.numel() else None,
                        "n_confident_fp": int((neg_p > 0.5).sum()) if neg_p.numel() else 0,
                        "matched_p_min": float(p[matched].min()) if matched.any() else None,
                        "match_hash": hashlib.sha1(repr(src_hashes).encode()).hexdigest()[:12],
                    }
            return loss

        criterion.loss_labels = loss_labels

    # ---------- trainer-facing API ----------
    def note_step_called(self):
        self.step_called = True

    def pre_step(self):
        if self.level != "full":
            return
        self._ce_first_done = False
        self.prev_params = [p.detach().clone() for p in self.model.parameters()]
        self.prev_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        self.step_called = False
        self.current_gn = {}

    def post_step(self, scaler):
        self.current_scale = float(scaler.get_scale())
        if self.level != "full":
            return
        raw_ce = self.current_losses.get("loss_ce")
        with torch.no_grad():
            update_norm = float(sum((p - q).float().norm() ** 2 for p, q in
                                    zip(self.model.parameters(), self.prev_params)) ** 0.5)
        self.prev_params = None
        nonfinite = (self.current_gn.get("nonfinite", False) or
                     any(isinstance(v, float) and not math.isfinite(v)
                         for v in self.current_losses.values() if v is not None))
        self.sink.write({"iter": self.iter, "kind": "full", "file_names": self.file_names,
                         "losses": self.current_losses, "gn": self.current_gn,
                         "update_norm": update_norm, "scaler_scale": self.current_scale,
                         "step_skipped": not self.step_called, "nonfinite": nonfinite,
                         "num_boxes": self.current_num_boxes, "ce_stats": self.ce_stats,
                         **self.rng_fp})
        self._maybe_trigger(raw_ce, update_norm, nonfinite)
        self.current_losses = {}

    def capture_losses(self, loss_dict):
        self.current_losses = {k: float(v) for k, v in loss_dict.items()}

    def periodic(self):
        if self.level != "full" or not self.pending_snaps:
            return
        for target, origin in sorted(self.pending_snaps):
            if target <= self.iter:
                self.pending_snaps.discard((target, origin))
                path = self.snap_dir / f"iter_{target:06d}_followup_{origin:06d}.pth"
                torch.save({k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}, path)
                self.events.write({"iter": self.iter, "kind": "snapshot", "path": str(path)})

    def _maybe_trigger(self, raw_ce, update_norm, nonfinite):
        it, gn = self.iter, self.current_gn.get("global_pre", 0.0)
        reason = None
        if nonfinite:
            reason = "nonfinite"
        elif not self.step_called:
            self.events.write({"iter": it, "kind": "alert", "type": "scaler_skip",
                               "scaler_scale": self.current_scale})
        else:
            if raw_ce is not None and raw_ce >= self.ALERT_CE:
                self.events.write({"iter": it, "kind": "alert", "loss_ce": raw_ce,
                                   "num_boxes": self.current_num_boxes})
                self.ce_high_run = self.ce_high_run + 1 if raw_ce >= self.TRIGGER_CE else 0
                if self.ce_high_run >= self.TRIGGER_CE_ITERS:
                    reason = "ce_sustained"
            if reason is None and len(self.grad_history) >= 50:
                med = sorted(self.grad_history)[len(self.grad_history) // 2]
                if med > 0 and gn > self.GRAD_SPIKE_FACTOR * med:
                    reason = "grad_spike"
        if self.force_trigger_iter is not None and it == self.force_trigger_iter:
            reason = "forced_acceptance"
        self.grad_history.append(gn)
        if reason is None or it < self.cool_until or self.trigger_count >= self.EVAL_BUDGET:
            return
        self.trigger_count += 1
        self.cool_until = it + self.COOLDOWN
        self.ce_high_run = 0
        event = {"iter": it, "kind": "trigger", "reason": reason, "raw_ce": raw_ce, "gn": gn,
                 "update_norm": update_norm, "scaler_scale": self.current_scale,
                 "num_boxes": self.current_num_boxes, "losses": dict(self.current_losses),
                 "gn_detail": dict(self.current_gn), "ce_stats": dict(self.ce_stats)}
        if self.prev_state is not None:
            path = self.snap_dir / f"iter_{it:06d}_prev.pth"
            torch.save(self.prev_state, path)
            event["snapshot_prev"] = str(path)
        hit = self.snap_dir / f"iter_{it:06d}_hit.pth"
        torch.save({k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}, hit)
        event["snapshot_hit"] = str(hit)
        for offset in self.SNAP_SCHEDULE:
            self.pending_snaps.add((it + offset, it))
        self.recent_hit = str(hit)
        self.events.write(event)
        if self.eval_count < self.EVAL_BUDGET:
            self.eval_count += 1
            self._launch_snapshot_eval(it, hit)

    def _launch_snapshot_eval(self, it, snapshot):
        script = Path(__file__).resolve().parents[2] / "scripts/eval_diag_snapshot.py"
        log = (self.dir / f"snapshot_eval_iter_{it:06d}.log").open("w")
        subprocess.Popen([sys.executable, "-B", str(script),
                          "--experiment", self.experiment_json,
                          "--checkpoint", str(snapshot),
                          "--output", str(snapshot) + ".eval.json",
                          "--max-images", "60"],
                         stdout=log, stderr=subprocess.STDOUT)
        self.events.write({"iter": it, "kind": "eval_launched", "snapshot": str(snapshot)})


def dataclasses_replace(options, **changes):
    import dataclasses
    return dataclasses.replace(options, **changes)
