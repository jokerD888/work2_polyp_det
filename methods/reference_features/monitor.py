"""Read-only optimizer/gradient evidence for the additional sampling branch."""
import hashlib
import json
from pathlib import Path
import torch
from polypbench.io import write_json


def adapter_named(model):
    return [(n, p) for n, p in model.named_parameters() if '.reference_features.' in n]


def native_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if '.reference_features.' not in name:
            digest.update(name.encode())
            digest.update(tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def optimizer_contract(model, optimizer):
    expected = {id(p): n for n, p in model.named_parameters() if p.requires_grad}
    actual, rows = {}, []
    for group in optimizer.param_groups:
        for parameter in group['params']:
            actual[id(parameter)] = actual.get(id(parameter), 0) + 1
            if '.reference_features.' in expected.get(id(parameter), ''):
                rows.append({'name': expected[id(parameter)], 'lr': group['lr'],
                    'weight_decay': group['weight_decay'], 'elements': parameter.numel()})
    if set(actual) != set(expected) or any(v != 1 for v in actual.values()):
        raise RuntimeError('Every trainable parameter must be discovered exactly once')
    count = sum(p.numel() for _, p in adapter_named(model))
    assert (count == 0) == (model.reference_features_variant == 'native')
    return {'status': 'passed', 'variant': model.reference_features_variant,
        'all_parameters_discovered_once': True, 'new_parameter_count': count,
        'new_parameter_groups': rows}


def snapshot_adapter(model):
    return {n: p.detach().clone() for n, p in adapter_named(model)}


def update_record(model, before):
    result = {}
    for name, parameter in adapter_named(model):
        grad = parameter.grad
        finite = grad is None or bool(torch.isfinite(grad).all())
        result[name] = {'gradient_present': grad is not None, 'gradient_finite': finite,
            'unscaled_after_clip_gradient_l2': float(grad.float().norm()) if grad is not None and finite else None,
            'update_l2': float((parameter.detach().float()-before[name].float()).norm()),
            'parameter_finite': bool(torch.isfinite(parameter).all())}
    return result


def make_hook(output):
    from detectron2.engine import HookBase
    class FeatureUpdateHook(HookBase):
        def before_train(self):
            write_json(Path(output)/'reference_optimizer_contract.json', optimizer_contract(self.trainer.model, self.trainer.optimizer))
            write_json(Path(output)/'reference_initialization.json', {'native_parameters_sha256': native_digest(self.trainer.model)})

        def before_step(self):
            self.observe = self.trainer.iter < 20 or self.trainer.iter % 275 == 0
            self.before = snapshot_adapter(self.trainer.model) if self.observe else None

        def after_step(self):
            if self.observe:
                record = {'iteration': self.trainer.iter,
                    'successful_updates': self.trainer._trainer.successful_updates,
                    'amp_scale': self.trainer._trainer.grad_scaler.get_scale(),
                    'adapter': update_record(self.trainer.model, self.before)}
                with (Path(output)/'reference_updates.jsonl').open('a') as stream:
                    stream.write(json.dumps(record, allow_nan=False)+'\n')
            self.before = None
    return FeatureUpdateHook()
