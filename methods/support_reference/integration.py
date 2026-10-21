"""Reversible, pinned native-forward attachment before DN concatenation.

The installed third-party source is never edited. Disabled calls use the
original bound method. The extra module is registered for normal state_dict
and parameter discovery, but is NOT enabled by an existing experiment config.
"""
import ast
import hashlib
import inspect
import math
from pathlib import Path
import textwrap
import types

import torch
from torch.nn import functional as F

from polypbench.analysis.support_reference_prototype import (
    SupportReferencePrototype, support_descriptor)


DECODER_SHA256 = '6e05a8413692b6467ba3f24230d97e003a1cbf04aa48935f87da1ee7e0d45e5a'
MODULE_NAME = 'support_reference'


def valid_canvas(batched_inputs, size_divisibility, device):
    """Only image geometry and mapper padding; no target/annotation access."""
    shapes = [tuple(x['image'].shape[-2:]) for x in batched_inputs]
    height, width = map(max, zip(*shapes))
    if size_divisibility > 1:
        height = math.ceil(height / size_divisibility) * size_divisibility
        width = math.ceil(width / size_divisibility) * size_divisibility
    result = torch.zeros((len(shapes), height, width), dtype=torch.bool, device=device)
    for i, (record, (h, w)) in enumerate(zip(batched_inputs, shapes)):
        padding = record.get('padding_mask')
        if padding is None:
            result[i, :h, :w] = True
        else:
            if tuple(padding.shape) != (h, w) or padding.dtype != torch.bool:
                raise ValueError('Mapper padding must be bool and match augmented image')
            result[i, :h, :w] = ~padding.to(device)
    return result


def compile_forward(original, replace):
    path = Path(inspect.getsourcefile(original.__func__))
    if hashlib.sha256(path.read_bytes()).hexdigest() != DECODER_SHA256:
        raise RuntimeError('Pinned native Decoder source has changed')
    tree = ast.parse(textwrap.dedent(inspect.getsource(original.__func__)))
    function = tree.body[0]
    positions = [i for i, node in enumerate(function.body)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
            t.id == 'tgt_mask' for t in node.targets)]
    if len(positions) != 1 or function.decorator_list:
        raise RuntimeError('Pinned initialization insertion point drift')
    function.body.insert(positions[0], ast.parse(
        'refpoint_embed = _support_reference_replace(outputs_mask, '
        'refpoint_embed_undetach, refpoint_embed, topk_proposals, tgt)').body[0])
    ast.fix_missing_locations(tree)
    namespace = dict(original.__func__.__globals__)
    namespace['_support_reference_replace'] = replace
    exec(compile(tree, '<support-reference-native-forward>', 'exec'), namespace)
    return types.MethodType(namespace[function.name], original.__self__)


class SupportReferenceAttachment:
    def __init__(self, model, enabled=False, capture=False):
        self.model = model
        self.decoder = model.sem_seg_head.predictor
        self.enabled, self.capture = enabled, capture
        self.last = {}
        self.canvas = None

    def _prepare(self, model, arguments):
        self.last = {}
        self.canvas = None
        if self.enabled:
            self.canvas = valid_canvas(arguments[0], model.size_divisibility, model.device)

    def _finish(self, model, arguments, result):
        self.canvas = None

    def replace(self, masks, encoder, native, top_ids, content):
        if self.canvas is None:
            raise RuntimeError('Attachment requires the full model forward for geometry')
        valid = F.interpolate(self.canvas[:, None].float(), size=masks.shape[-2:],
                              mode='nearest-exact')[:, 0].bool()
        result = self.adapter(masks, encoder, native, valid)
        if self.capture:
            features, eligible = support_descriptor(masks, encoder, native, valid)
            self.last = {'native': native.detach(), 'actual': result.detach(),
                'encoder': encoder.detach(), 'top_ids': top_ids.detach(),
                'content': content.detach(), 'features': features, 'eligible': eligible,
                'valid_lattice': valid.detach(), 'canvas_hw': tuple(self.canvas.shape[-2:])}
        return result

    def _forward(self, *args, **kwargs):
        return (self.compiled if self.enabled else self.original)(*args, **kwargs)

    def __enter__(self):
        decoder = self.decoder
        if MODULE_NAME in decoder._modules or 'forward' in decoder.__dict__:
            raise RuntimeError('Refuse nested attachment or existing forward override')
        if not (decoder.two_stage and decoder.initial_pred and not decoder.learn_tgt
                and decoder.initialize_box_type == 'mask2box' and decoder.num_queries == 300):
            raise RuntimeError('Unsupported native configuration')
        self.original = decoder.forward
        self.compiled = compile_forward(self.original, self.replace)
        self.adapter = SupportReferencePrototype().to(self.model.device)
        # New modules do not inherit the parent's current eval state automatically.
        self.adapter.train(decoder.training)
        decoder.add_module(MODULE_NAME, self.adapter)
        decoder.forward = self._forward
        self.pre = self.model.register_forward_pre_hook(self._prepare)
        self.post = self.model.register_forward_hook(self._finish, always_call=True)
        return self

    def __exit__(self, *exc):
        self.pre.remove()
        self.post.remove()
        del self.decoder.forward
        delattr(self.decoder, MODULE_NAME)
        self.canvas = None
        self.last = {}


def load_native_checkpoint(model, native_state):
    """Allow exactly the registered new adapter keys, no other missing weights."""
    expected = {k for k in model.state_dict() if '.support_reference.' in k}
    if not expected or set(native_state) & expected:
        raise RuntimeError('Expected a native checkpoint without adapter keys')
    missing, unexpected = model.load_state_dict(native_state, strict=False)
    if set(missing) != expected or unexpected:
        raise RuntimeError(f'Unexpected checkpoint mismatch: missing={missing}, unexpected={unexpected}')
    return sorted(expected)
