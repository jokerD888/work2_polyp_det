from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Options:
    variant: str = 'box'
    weight: float = 0.1
    temperature: float = 0.1
    points: int = 32
    min_points: int = 2
    purity: float = 0.95
    core_fraction: float = 0.05
    ring_fraction: float = 0.20
    seed: int = 42

    def validate(self):
        if self.variant not in ('off', 'box', 'mask_priority'):
            raise ValueError('Unknown local contrast variant')
        for key in ('weight', 'temperature', 'purity', 'core_fraction', 'ring_fraction'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError('Invalid local contrast '+key)
        if self.purity > 1 or not 2 <= self.min_points <= self.points:
            raise ValueError('Require purity<=1 and 2<=min_points<=points')
        if any(type(getattr(self, k)) is not int for k in ('points', 'min_points', 'seed')):
            raise ValueError('Point budgets and seed must be integers')
        return self


def validate(experiment):
    experiment['local_contrast'] = asdict(Options(**experiment['local_contrast']).validate())
    if any(k in experiment for k in ('reference_features', 'support_reference', 'mssl_scope',
                                     'quality_alignment', 'box_regression', 'gradient_probe_period')):
        raise ValueError('Overlapping research routes are not supported')
    if not experiment['mssl']['enabled'] or experiment['mssl']['support'] != 'mask':
        raise ValueError('L1/L2 require the fixed MSSL V1')
    if experiment['runtime']['num_gpus'] != 1 or experiment['runtime']['micro_batch'] != 1:
        raise ValueError('This protocol uses one GPU and microbatch1')
    return experiment
