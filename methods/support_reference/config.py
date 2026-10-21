"""Fixed controlled variants, with no threshold/coefficient search interface."""
VARIANTS = ('native', 'blend05', 'geometry', 'support')


def validate(experiment):
    options = experiment.get('support_reference', {})
    if set(options) != {'variant'} or options['variant'] not in VARIANTS:
        raise ValueError('support_reference requires exactly a fixed variant')
    if any(k in experiment for k in ('quality_alignment', 'box_regression', 'mssl_scope', 'gradient_probe_period')):
        raise ValueError('Overlapping research routes are not supported')
    if experiment['runtime'].get('num_gpus') != 1 or experiment['runtime'].get('micro_batch') != 1:
        raise ValueError('Validated protocol is one GPU, microbatch1')
    if experiment['mssl']['enabled']:
        m = experiment['mssl']
        if m['support'] != 'mask' or m['weight'] != 1 or m.get('loss_form', 'cdf') != 'cdf':
            raise ValueError('Combination must retain unchanged MSSL V1')
    return experiment
