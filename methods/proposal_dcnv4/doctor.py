"""Explicit future P0 CUDA gate. Not executed by imports or CPU checking."""
import argparse
import datetime as dt
import sys


def main():
    parser = argparse.ArgumentParser(description="P0 official DCNv4 CUDA forward/backward gate")
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-gpu", action="store_true", help="Explicitly run only after the active training has ended")
    args = parser.parse_args()
    if not args.allow_gpu:
        parser.error("GPU execution is opt-in: wait for active training, then pass --allow-gpu")
    import torch
    from polypbench.io import write_json
    from .config import DCNv4Options
    from .dcnv4_adapter import DCNv4Residual
    report = {"gate": "P0_OPERATOR_ONLY", "passed": False,
              "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), "python": sys.version,
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "P1_full_model": "NOT_RUN", "P2_short_training": "NOT_RUN"}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.manual_seed(29)
        report["gpu"] = torch.cuda.get_device_name()
        report["compute_capability"] = torch.cuda.get_device_capability()
        adapter = DCNv4Residual(DCNv4Options(enabled=True)).cuda()
        report["backend"] = adapter.backend_identity
        report["unique_parameter_count"] = sum(p.numel() for p in adapter.parameters())
        if report["unique_parameter_count"] != 160368:
            raise AssertionError("Official default adapter parameter count differs from the pinned 160368")
        initial = torch.randn(2, 256, 7, 9, device="cuda")
        initial_delta = (adapter(initial) - initial).abs().max().item()
        report["initial_max_feature_delta"] = initial_delta
        # Exercise trained/nonzero aggregation weights as well as initialization.
        with torch.no_grad():
            adapter.dcn.offset_mask.weight.normal_(0, .002)
            adapter.dcn.offset_mask.bias.normal_(0, .01)
        records = []
        for batch, h, w in ((1, 7, 9), (2, 8, 12)):
            x = torch.randn(batch, 256, h, w, device="cuda", requires_grad=True)
            for amp in (False, True):
                adapter.zero_grad(set_to_none=True)
                x.grad = None
                with torch.autocast("cuda", enabled=amp):
                    y = adapter(x)
                    loss = y.square().mean()
                loss.backward()
                gradients = [p.grad for p in adapter.parameters() if p.grad is not None]
                if not torch.isfinite(y).all() or not torch.isfinite(x.grad).all():
                    raise FloatingPointError("Non-finite operator output/input gradient")
                if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                    raise FloatingPointError("Non-finite or absent operator parameter gradients")
                records.append({"shape": list(x.shape), "amp": amp,
                                "input_grad_norm": x.grad.norm().item(),
                                "parameter_grad_norm": sum(g.float().square().sum().item() for g in gradients) ** .5})
        # Directional finite difference, input and offset/mask weight separately.
        # This is FP32 finite difference, not double-precision torch.gradcheck.
        x = torch.randn(1, 256, 5, 7, device="cuda", requires_grad=True)
        probe = torch.randn_like(x)
        loss = (adapter(x) * probe).sum() / probe.numel()
        grad_x, grad_w = torch.autograd.grad(loss, (x, adapter.dcn.offset_mask.weight))
        finite_difference = []
        for name, tensor, gradient in (("input", x, grad_x), ("offset_mask.weight", adapter.dcn.offset_mask.weight, grad_w)):
            direction = torch.randn_like(tensor)
            direction /= direction.square().mean().sqrt()
            expected = (gradient * direction).sum().item()
            epsilon = 1e-3 if name == "input" else 1e-4
            with torch.no_grad():
                before = tensor.clone()
                try:
                    tensor.copy_(before + epsilon * direction)
                    plus = ((adapter(x) * probe).sum() / probe.numel()).item()
                    tensor.copy_(before - epsilon * direction)
                    minus = ((adapter(x) * probe).sum() / probe.numel()).item()
                finally:
                    tensor.copy_(before)
            measured = (plus - minus) / (2 * epsilon)
            difference = abs(measured - expected)
            passed = difference <= 1e-4 + .1 * abs(expected)
            finite_difference.append({"tensor": name, "analytical": expected,
                                      "finite_difference": measured, "epsilon": epsilon,
                                      "atol": 1e-4, "rtol": .1, "passed": passed})
            if not passed:
                raise AssertionError(f"Directional finite difference failed: {name}")
        report.update(passed=True, shapes=records, finite_difference=finite_difference)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(args.output, report)


if __name__ == "__main__":
    main()
