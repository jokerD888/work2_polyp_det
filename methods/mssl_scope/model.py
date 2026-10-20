from detectron2.modeling import META_ARCH_REGISTRY
from maskdino.modeling.criterion import SetCriterion
from methods.mssl.model import MSSLMaskDINO
from methods.mssl.loss import matched_loss


class ScopeCriterion(SetCriterion):
    """Identity routing uses the native match of each ordinary decoder output."""
    def forward(self, outputs, targets, mask_dict=None):
        if not outputs.get('aux_outputs'):
            raise ValueError('A penultimate ordinary decoder prediction is required')
        self._routes = {'final': outputs, 'aux': outputs['aux_outputs'][-1]}
        self._parts = {}; self.route_calls = {'final': 0, 'aux': 0}
        self.last_scope_stats = {}
        try:
            losses = super().forward(outputs, targets, mask_dict)
            if self.route_calls != {'final': 1, 'aux': 1}:
                raise RuntimeError(f'Wrong MSSL routes: {self.route_calls}')
            final = self._parts['final']['loss']
            auxiliary = self._parts['aux']['loss']
            total = self.final_weight * final + self.aux_weight * auxiliary
            losses['loss_mssl'] = total
            self.last_diagnostics = self._parts['final']['diagnostics']
            self.last_scope_stats = {'final_raw': float(final.detach()), 'aux_raw': float(auxiliary.detach()),
                'final_weight': self.final_weight, 'aux_weight': self.aux_weight,
                'weighted_total': float(total.detach())}
            callback = getattr(self, 'probe_callback', None)
            if callback is not None:
                callback(self, losses, self._parts)
            return losses
        finally:
            self._routes = {}; self._parts = {}

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        result = super().get_loss(loss, outputs, targets, indices, num_masks)
        if loss == 'boxes':
            for name, route in self._routes.items():
                if outputs is route:
                    self.route_calls[name] += 1
                    # Also measure La in E, without adding it to the objective.
                    value, diag = matched_loss(outputs, targets, indices, self.options)
                    self._parts[name] = {'loss': value, 'diagnostics': diag, 'outputs': outputs,
                        'indices': indices, 'native_boxes': result}
        return result


@META_ARCH_REGISTRY.register()
class ScopeMaskDINO(MSSLMaskDINO):
    @classmethod
    def from_config(cls, cfg):
        params = super().from_config(cfg)
        native = params['criterion']
        names = ('num_classes', 'matcher', 'weight_dict', 'eos_coef', 'losses', 'num_points',
                 'oversample_ratio', 'importance_sample_ratio', 'dn', 'dn_losses', 'panoptic_on', 'semantic_ce_loss')
        criterion = ScopeCriterion(**{k: getattr(native, k) for k in names})
        criterion.options = native.options
        criterion.final_weight = cfg.MODEL.MSSL_SCOPE_FINAL_WEIGHT
        criterion.aux_weight = cfg.MODEL.MSSL_SCOPE_AUX_WEIGHT
        params['criterion'] = criterion
        return params
