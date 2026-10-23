VARIANTS = ('native', 'mask_extra', 'encoder_extra')


def validate(experiment):
    options = experiment.get('reference_features', {})
    if set(options) != {'variant'} or options['variant'] not in VARIANTS:
        raise ValueError('reference_features requires exactly a fixed variant')
    if any(k in experiment for k in ('support_reference', 'quality_alignment', 'box_regression',
                                     'mssl_scope', 'gradient_probe_period')):
        raise ValueError('Overlapping research routes are not supported')
    if experiment['runtime'].get('num_gpus') != 1 or experiment['runtime'].get('micro_batch') != 1:
        raise ValueError('Validated protocol is one GPU, microbatch1')
    if experiment['mssl']['enabled']:
        raise ValueError('This standalone decision stage keeps MSSL disabled')
    return experiment
