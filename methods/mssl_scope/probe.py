"""Weighted shared bbox-head gradients, common measurement scale, no .grad writes."""
import json
from pathlib import Path
import torch


class ScopeGradientProbe:
    def __init__(self, model, output, period=275):
        self.parameters = [p for n, p in model.named_parameters() if 'bbox_embed' in n and p.requires_grad]
        if not self.parameters:
            raise RuntimeError('No shared native bbox head found')
        self.output = Path(output); self.period = period

    def callback(self, iteration, micro_index):
        if micro_index != 0 or (iteration not in (0, 10) and iteration % self.period):
            return None

        def probe(criterion, losses, parts):
            weight = criterion.weight_dict
            final_native = sum(v * weight[k] for k, v in parts['final']['native_boxes'].items())
            all_native = sum(v * weight[k] for k, v in losses.items()
                             if k.startswith(('loss_bbox', 'loss_giou')) and k in weight)
            values = {'gf': parts['final']['loss'] * weight['loss_mssl'],
                      'ga': parts['aux']['loss'] * weight['loss_mssl'],
                      'total': losses['loss_mssl'] * weight['loss_mssl'],
                      'native_final': final_native, 'native_all': all_native}
            attempts = []
            for scale in (64., 8., 1., .125, .015625):
                gradients = {}
                for name, value in values.items():
                    grads = torch.autograd.grad(value * scale, self.parameters, retain_graph=True, allow_unused=True)
                    gradients[name] = torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().reshape(-1) / scale
                        for p, g in zip(self.parameters, grads)])
                finite = all(bool(torch.isfinite(g).all()) for g in gradients.values())
                attempts.append({'scale': scale, 'finite': finite})
                if finite:
                    break
            else:
                raise FloatingPointError(f'Nonfinite bbox diagnostic: {attempts}')
            norms = {k: float(g.norm()) for k, g in gradients.items()}
            def cosine(a, b):
                denominator = norms[a] * norms[b]
                return float(torch.dot(gradients[a], gradients[b]) / denominator) if denominator > 1e-20 else None
            row = {'iteration': iteration, 'micro_index': micro_index, 'scale': scale, 'attempts': attempts,
                'norms': norms, 'cosines': {a+'__'+b: cosine(a,b) for a,b in
                    (('gf','ga'),('gf','native_final'),('ga','native_final'),('total','native_final'),
                     ('gf','native_all'),('ga','native_all'),('total','native_all'))},
                'norm_ratios': {a+'__'+b: norms[a]/norms[b] if norms[b]>1e-20 else None
                    for a in ('gf','ga','total') for b in ('native_final','native_all')},
                'weighted_losses': {k: float(v.detach()) for k,v in values.items()},
                'coefficients': [criterion.final_weight, criterion.aux_weight],
                'head_linearity_relative_residual': float((gradients['total'] - criterion.final_weight * gradients['gf'] -
                    criterion.aux_weight * gradients['ga']).norm()) / max(norms['total'], 1e-20),
                'diagnostics': {k: p['diagnostics'] for k,p in parts.items()},
                'scope': 'shared bbox head, first microbatch, weighted before AMP/clip/accumulation; common measurement-only scale, FP16 precision limited'}
            self.output.parent.mkdir(parents=True, exist_ok=True)
            with self.output.open('a', encoding='utf-8') as f:
                f.write(json.dumps(row, allow_nan=False)+'\n')
        return probe
