from dataclasses import replace
from pathlib import Path
import socket
import tempfile
import unittest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from methods.scda_v3.config import V3Options
from methods.scda_v3.aux_head import clone_aux_head, predict_selected
from methods.scda_v3.assigner import assign
from methods.scda_v3.loss import auxiliary_loss
from methods.scda_proposal.scda_loss import scda_losses
from methods.scda_proposal.structures import ImageAssignment, TaskAssignment
from .fixtures import fixture


def head():
    result = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU(),
                                  torch.nn.Linear(16, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4))
    return result


def ddp_case(model, rank, empty_global=False):
    payload, target = fixture(shapes=((4, 4),), gt_count=0 if rank == 1 or empty_global else 1)
    payload["output_memory"] = torch.ones_like(payload["output_memory"])
    tokens = torch.tensor([5, 6]) if len(target.labels) else torch.empty(0, dtype=torch.long)
    task = TaskAssignment(tokens, torch.zeros_like(tokens))
    assignment = ImageAssignment(task, task, torch.ones(16, 4))
    return auxiliary_loss(model, payload, [target], [assignment], V3Options())


def gloo_worker(rank, port, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    try:
        torch.manual_seed(7)
        model = DistributedDataParallel(head())
        results = {}
        for empty in (False, True):
            model.zero_grad(set_to_none=True)
            losses, counts = ddp_case(model, rank, empty)
            losses["loss_v3_box"].backward()
            results[empty] = {"grads": [p.grad.clone() for p in model.parameters()], "counts": counts}
        torch.save(results, Path(directory)/f"{rank}.pt")
    finally:
        dist.destroy_process_group()


class HeadLossTests(unittest.TestCase):
    def test_independent_storage_predictions_prior_and_rng(self):
        native = head(); state = torch.get_rng_state().clone()
        auxiliary = clone_aux_head(native)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        payload, _ = fixture(); p = torch.tensor([5, 6, 40])
        torch.testing.assert_close(predict_selected(auxiliary, payload, 0, p),
                                   (native(payload["output_memory"][0, p]) + payload["proposal_unsigmoid"][0, p]).sigmoid(),
                                   rtol=0, atol=0)
        self.assertTrue(all(a.data_ptr() != b.data_ptr() for a, b in zip(native.parameters(), auxiliary.parameters())))

    def test_aux_loss_matches_old_objective_given_equal_box_predictions(self):
        payload, target = fixture(); auxiliary = head()
        payload["raw_box_sigmoid"] = (auxiliary(payload["output_memory"]) + payload["proposal_unsigmoid"]).sigmoid()
        options = V3Options(); assignments = assign(payload, [target], options)
        actual, _ = auxiliary_loss(auxiliary, payload, [target], assignments, options)
        old = scda_losses(payload, [target], assignments, options.legacy(), 5., 2.)
        torch.testing.assert_close(actual["loss_v3_box"], old.losses["loss_scda_box"], rtol=1e-6, atol=1e-7)

    def test_only_selected_tokens_use_mlp_and_gradients_do_not_touch_native_heads(self):
        payload, target = fixture(); native = head(); auxiliary = clone_aux_head(native)
        payload["raw_box_sigmoid"] = (native(payload["output_memory"]) + payload["proposal_unsigmoid"]).sigmoid()
        options = V3Options(); assignments = assign(payload, [target], options)
        seen = []; handle = auxiliary.register_forward_pre_hook(lambda m, args: seen.append(args[0].shape[0]))
        losses, _ = auxiliary_loss(auxiliary, payload, [target], assignments, options)
        handle.remove(); losses["loss_v3_box"].backward()
        self.assertEqual(seen, [assignments[0].reg.token_indices.numel()])
        self.assertTrue(all(p.grad is None for p in native.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in auxiliary.parameters()))
        self.assertGreater(float(payload["output_memory"].grad.abs().sum()), 0)
        self.assertIsNone(payload["raw_cls"].grad)

    def test_empty_and_zero_lambda_have_finite_connected_zero_for_all_aux_params(self):
        for empty, coefficient in ((True, .05), (False, 0.)):
            payload, target = fixture(gt_count=0 if empty else 1); auxiliary = head()
            options = V3Options(lambda_aux=coefficient); assignments = assign(payload, [target], options)
            losses, _ = auxiliary_loss(auxiliary, payload, [target], assignments, options)
            losses["loss_v3_box"].backward()
            self.assertEqual(float(losses["loss_v3_box"]), 0)
            for p in auxiliary.parameters():
                self.assertIsNotNone(p.grad); self.assertTrue(torch.equal(p.grad, torch.zeros_like(p.grad)))

    def test_nonfinite_selected_features_or_head_raise(self):
        payload, target = fixture(); auxiliary = head(); options = V3Options()
        assignments = assign(payload, [target], options)
        payload["output_memory"].data[0, assignments[0].reg.token_indices[0]] = float("nan")
        with self.assertRaises(FloatingPointError): auxiliary_loss(auxiliary, payload, [target], assignments, options)

    def test_cpu_autocast_prediction_and_backward_are_finite(self):
        payload, target = fixture(); auxiliary = head(); options = V3Options()
        assignments = assign(payload, [target], options)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            losses, _ = auxiliary_loss(auxiliary, payload, [target], assignments, options)
        losses["loss_v3_box"].backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in auxiliary.parameters()))

    def test_ddp_global_denominator_empty_rank_and_global_empty(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(gloo_worker, args=(port, directory), nprocs=2, join=True)
            a, b = (torch.load(Path(directory)/f"{rank}.pt", weights_only=True) for rank in (0, 1))
        torch.manual_seed(7); reference = head()
        loss, _ = ddp_case(reference, 0); loss["loss_v3_box"].backward()
        for ga, gb, parameter in zip(a[False]["grads"], b[False]["grads"], reference.parameters()):
            torch.testing.assert_close(ga, gb)
            torch.testing.assert_close(ga, parameter.grad)
        self.assertEqual(float(a[False]["counts"][0]), 1)
        self.assertTrue(all(torch.equal(g, torch.zeros_like(g)) for g in a[True]["grads"]))


if __name__ == "__main__": unittest.main()
