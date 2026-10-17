"""Independent native-shaped MLP; only selected memory tokens are forwarded."""
import copy
import torch


def clone_aux_head(native_head):
    head = copy.deepcopy(native_head)
    for old, new in zip(native_head.parameters(), head.parameters()):
        if old is new or old.data_ptr() == new.data_ptr():
            raise AssertionError("Aux head shares parameter storage with native head")
    return head


def predict_selected(head, payload, batch_index, tokens):
    memory = payload["output_memory"][batch_index, tokens]
    prior = payload["proposal_unsigmoid"][batch_index, tokens].detach()
    if not torch.isfinite(memory).all() or not torch.isfinite(prior).all():
        raise FloatingPointError("Non-finite selected V3 memory/proposal")
    # Exactly the native residual parameterization, including the unsigmoid
    # prior. An empty selection still traverses ALL MLP parameters for DDP.
    result = (head(memory) + prior).sigmoid()
    if not torch.isfinite(result).all():
        raise FloatingPointError("Non-finite auxiliary box prediction")
    return result
