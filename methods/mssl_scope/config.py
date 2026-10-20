VARIANTS = {'E': (2., 0.), 'F': (1., 1.)}


def validate(experiment):
    o = experiment.get('mssl_scope', {})
    if set(o) != {'variant', 'final_weight', 'aux_weight'} or o.get('variant') not in VARIANTS:
        raise ValueError('Only the fixed E/F protocol is supported')
    if (o['final_weight'], o['aux_weight']) != VARIANTS[o['variant']]:
        raise ValueError('E must be 2Lf; F must be Lf+La')
    m = experiment['mssl']
    if not m['enabled'] or m['weight'] != 1. or m['support'] != 'mask' or m.get('loss_form', 'cdf') != 'cdf':
        raise ValueError('Requires unchanged unit-weight Mask-CDF V1')
    if any(k in experiment for k in ('quality_alignment', 'box_regression', 'gradient_probe_period')):
        raise ValueError('No overlapping method or gradient probe')
    return experiment
