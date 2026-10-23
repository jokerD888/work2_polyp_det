"""Keep native geometry; add ordinary-query feature residual at layer0 CA.

No third-party file is edited. The native cross-attention result is computed
unchanged, then an independent cloned sampler + zero-output projection adds a
feature residual only for the final 300 ordinary queries. H1/H2 differ solely
in the extra sampler's reference coordinates. Native query positions are shared.
"""
import ast
import copy
import hashlib
import inspect
from pathlib import Path
import textwrap
import types

import torch
from torch import nn


DECODER_SHA256 = '6e05a8413692b6467ba3f24230d97e003a1cbf04aa48935f87da1ee7e0d45e5a'
MODULE_NAME = 'reference_features'


def scaled_encoder_reference(encoder_logits, valid_ratios):
    """B,Q,4 (same native coordinate convention) -> B,Q,L,4."""
    return encoder_logits.detach().sigmoid()[:, :, None] * torch.cat(
        [valid_ratios, valid_ratios], -1)[:, None]


class ExtraRead(nn.Module):
    def __init__(self, original_sampler, dimension):
        super().__init__()
        self.sampler = copy.deepcopy(original_sampler)
        # New branch construction must not shift native initialization/LSJ RNG.
        with torch.random.fork_rng(devices=[]):
            self.projection = nn.Linear(dimension, dimension, bias=False)
            nn.init.zeros_(self.projection.weight)

    def forward(self, query, reference, memory, shapes, starts, padding):
        return self.projection(self.sampler(query, reference, memory, shapes, starts, padding))


def compile_forward(original, callback):
    source = Path(inspect.getsourcefile(original.__func__))
    if hashlib.sha256(source.read_bytes()).hexdigest() != DECODER_SHA256:
        raise RuntimeError('Pinned native predictor source changed')
    tree = ast.parse(textwrap.dedent(inspect.getsource(original.__func__)))
    function = tree.body[0]
    indices = [i for i, node in enumerate(function.body) if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'tgt_mask' for t in node.targets)]
    if len(indices) != 1 or function.decorator_list:
        raise RuntimeError('Native capture location drift')
    function.body.insert(indices[0], ast.parse('_reference_features_capture('
        'refpoint_embed_undetach, refpoint_embed, topk_proposals, tgt, valid_ratios)').body[0])
    ast.fix_missing_locations(tree)
    namespace = dict(original.__func__.__globals__)
    namespace['_reference_features_capture'] = callback
    exec(compile(tree, '<reference-features-native-forward>', 'exec'), namespace)
    return types.MethodType(namespace[function.name], original.__self__)


class ReferenceFeaturesAttachment:
    def __init__(self, model, variant, enabled=True, capture=False):
        if variant not in ('mask_extra', 'encoder_extra'):
            raise ValueError('Unknown extra-read variant')
        self.model, self.variant = model, variant
        self.decoder = model.sem_seg_head.predictor
        self.enabled, self.capture = enabled, capture
        self.last = {}
        self.context = None

    def _capture(self, encoder, native, top_ids, content, valid_ratios):
        if encoder.shape != native.shape or encoder.shape[1:] != (300, 4):
            raise RuntimeError('Expected unchanged Top300 ordinary references')
        self.context = scaled_encoder_reference(encoder, valid_ratios)
        if self.capture:
            self.last = {'native': native.detach(), 'actual': native.detach(),
                'encoder': encoder.detach(), 'top_ids': top_ids.detach(),
                'content': content.detach(), 'valid_ratios': valid_ratios.detach()}

    def _forward(self, *args, **kwargs):
        self.context = None
        self.last = {}
        try:
            return (self.compiled if self.enabled else self.original)(*args, **kwargs)
        finally:
            self.context = None

    def _extra(self, module, arguments, output):
        if not self.enabled:
            return output
        if self.context is None:
            raise RuntimeError('Extra feature reading requires native predictor context')
        query, reference, memory, shapes, starts, padding = arguments
        count = query.shape[1]
        if count < 300:
            raise RuntimeError('Missing ordinary queries')
        reference_extra = reference[:, -300:] if self.variant == 'mask_extra' else self.context
        if reference_extra.shape != reference[:, -300:].shape:
            raise RuntimeError('Reference batch/level mismatch')
        residual = self.adapter(query[:, -300:], reference_extra, memory, shapes, starts, padding)
        ordinary = output[:, -300:] + residual
        result = torch.cat([output[:, :-300], ordinary], 1) if count > 300 else ordinary
        if self.capture:
            self.last.update(main_reference=reference[:, -300:].detach(),
                extra_reference=reference_extra.detach(), dn_pad=count-300,
                dn_cross_unchanged=torch.equal(output[:, :-300], result[:, :-300]),
                residual_l2=residual.detach().float().norm(), extra_calls=1)
        return result

    def __enter__(self):
        decoder = self.decoder
        if 'forward' in decoder.__dict__ or not (decoder.two_stage and decoder.initial_pred
                and not decoder.learn_tgt and decoder.initialize_box_type == 'mask2box'
                and decoder.num_queries == 300):
            raise RuntimeError('Unsupported/nested native predictor attachment')
        layer = decoder.decoder.layers[0]
        if MODULE_NAME in layer._modules or layer.cross_attn._forward_hooks:
            raise RuntimeError('Unsupported/nested first-layer attention attachment')
        self.original = decoder.forward
        self.compiled = compile_forward(self.original, self._capture)
        self.adapter = ExtraRead(layer.cross_attn, decoder.hidden_dim).to(self.model.device)
        self.adapter.train(decoder.training)
        layer.add_module(MODULE_NAME, self.adapter)
        self.first_layer = layer
        self.handle = layer.cross_attn.register_forward_hook(self._extra)
        decoder.forward = self._forward
        return self

    def __exit__(self, *exc):
        self.handle.remove()
        del self.decoder.forward
        delattr(self.first_layer, MODULE_NAME)
        self.context = None
        self.last = {}
