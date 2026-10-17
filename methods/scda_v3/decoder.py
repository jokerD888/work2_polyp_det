# Derived from IDEA-Research/MaskDINO, commit
# 3831d8514a3728535ace8d4ecc7d28044c42dd14.
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0; see LICENSE.upstream.
# V3 additions: independent subclass, training-only payload; native forward
# statements (including the original torch.topk call) are preserved below.
import torch
from detectron2.config import configurable
from detectron2.structures import BitMasks
from maskdino.modeling.transformer_decoder.maskdino_decoder import (
    MaskDINODecoder, TRANSFORMER_DECODER_REGISTRY,
)
from maskdino.utils import box_ops
from maskdino.utils.utils import gen_encoder_output_proposals, inverse_sigmoid

from .config import V3Options
from .payload import make_payload


@TRANSFORMER_DECODER_REGISTRY.register()
class SCDAv3Decoder(MaskDINODecoder):
    @configurable
    def __init__(self, *, native_kwargs, scda_options):
        super().__init__(**native_kwargs)
        self.scda_options = scda_options
        if scda_options.enabled and not self.two_stage:
            raise ValueError("F-v1 requires TWO_STAGE=True")
        if scda_options.enabled and self.semantic_ce_loss:
            raise ValueError("F-v1 does not implement softmax/background CE adaptation")

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        return {
            "native_kwargs": super().from_config(cfg, in_channels, mask_classification),
            "scda_options": V3Options.from_cfg(cfg),
        }

    def forward(self, x, mask_features, masks, targets=None):
        """
        :param x: input, a list of multi-scale feature
        :param mask_features: is the per-pixel embeddings with resolution 1/4 of the original image,
        obtained by fusing backbone encoder encoded features. This is used to produce binary masks.
        :param masks: mask in the original image
        :param targets: used for denoising training
        """
        if not (self.training and self.scda_options.enabled):
            return super().forward(x, mask_features, masks, targets=targets)
        assert len(x) == self.num_feature_levels
        device = x[0].device
        size_list = []
        # disable mask, it does not affect performance
        enable_mask = 0
        if masks is not None:
            for src in x:
                if src.size(2) % 32 or src.size(3) % 32:
                    enable_mask = 1
        if enable_mask == 0:
            masks = [torch.zeros((src.size(0), src.size(2), src.size(3)), device=src.device, dtype=torch.bool) for src in x]
        src_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for i in range(self.num_feature_levels):
            idx=self.num_feature_levels-1-i
            bs, c , h, w=x[idx].shape
            size_list.append(x[i].shape[-2:])
            spatial_shapes.append(x[idx].shape[-2:])
            src_flatten.append(self.input_proj[idx](x[idx]).flatten(2).transpose(1, 2))
            mask_flatten.append(masks[i].flatten(1))
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        predictions_class = []
        predictions_mask = []
        if self.two_stage:
            output_memory, output_proposals = gen_encoder_output_proposals(src_flatten, mask_flatten, spatial_shapes)
            output_memory = self.enc_output_norm(self.enc_output(output_memory))
            enc_outputs_class_unselected = self.class_embed(output_memory)
            enc_outputs_coord_unselected = self._bbox_embed(
                output_memory) + output_proposals  # (bs, \sum{hw}, 4) unsigmoid
            topk = self.num_queries
            topk_proposals = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]
            refpoint_embed_undetach = torch.gather(enc_outputs_coord_unselected, 1,
                                                   topk_proposals.unsqueeze(-1).repeat(1, 1, 4))  # unsigmoid
            refpoint_embed = refpoint_embed_undetach.detach()

            tgt_undetach = torch.gather(output_memory, 1,
                                  topk_proposals.unsqueeze(-1).repeat(1, 1, self.hidden_dim))  # unsigmoid

            outputs_class, outputs_mask = self.forward_prediction_heads(tgt_undetach.transpose(0, 1), mask_features)
            tgt = tgt_undetach.detach()
            if self.learn_tgt:
                tgt = self.query_feat.weight[None].repeat(bs, 1, 1)
            interm_outputs=dict()
            interm_outputs['pred_logits'] = outputs_class
            interm_outputs['pred_boxes'] = refpoint_embed_undetach.sigmoid()
            interm_outputs['pred_masks'] = outputs_mask

            if self.initialize_box_type != 'no':
                # convert masks into boxes to better initialize box in the decoder
                assert self.initial_pred
                flaten_mask = outputs_mask.detach().flatten(0, 1)
                h, w = outputs_mask.shape[-2:]
                if self.initialize_box_type == 'bitmask':  # slower, but more accurate
                    refpoint_embed = BitMasks(flaten_mask > 0).get_bounding_boxes().tensor.to(device)
                elif self.initialize_box_type == 'mask2box':  # faster conversion
                    refpoint_embed = box_ops.masks_to_boxes(flaten_mask > 0).to(device)
                else:
                    assert NotImplementedError
                refpoint_embed = box_ops.box_xyxy_to_cxcywh(refpoint_embed) / torch.as_tensor([w, h, w, h],
                                                                                              dtype=torch.float).to(device)
                refpoint_embed = refpoint_embed.reshape(outputs_mask.shape[0], outputs_mask.shape[1], 4)
                refpoint_embed = inverse_sigmoid(refpoint_embed)
            scda_payload = make_payload(
                enc_outputs_class_unselected, enc_outputs_coord_unselected,
                output_proposals, topk_proposals, spatial_shapes, level_start_index,
                mask_flatten, output_memory,
                refpoint_embed.sigmoid() if self.scda_options.diagnostics else None,
                diagnostics=self.scda_options.diagnostics,
                init_from_mask=self.initialize_box_type != 'no',
            )
        elif not self.two_stage:
            tgt = self.query_feat.weight[None].repeat(bs, 1, 1)
            refpoint_embed = self.query_embed.weight[None].repeat(bs, 1, 1)

        tgt_mask = None
        mask_dict = None
        if self.dn != "no" and self.training:
            assert targets is not None
            input_query_label, input_query_bbox, tgt_mask, mask_dict = \
                self.prepare_for_dn(targets, None, None, x[0].shape[0])
            if mask_dict is not None:
                tgt=torch.cat([input_query_label, tgt],dim=1)

        # direct prediction from the matching and denoising part in the begining
        if self.initial_pred:
            outputs_class, outputs_mask = self.forward_prediction_heads(tgt.transpose(0, 1), mask_features, self.training)
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
        if self.dn != "no" and self.training and mask_dict is not None:
            refpoint_embed=torch.cat([input_query_bbox,refpoint_embed],dim=1)

        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=src_flatten.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=None,
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=tgt_mask
        )
        for i, output in enumerate(hs):
            outputs_class, outputs_mask = self.forward_prediction_heads(output.transpose(0, 1), mask_features, self.training or (i == len(hs)-1))
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        # iteratively box prediction
        if self.initial_pred:
            out_boxes = self.pred_box(references, hs, refpoint_embed.sigmoid())
            assert len(predictions_class) == self.num_layers + 1
        else:
            out_boxes = self.pred_box(references, hs)
        if mask_dict is not None:
            predictions_mask=torch.stack(predictions_mask)
            predictions_class=torch.stack(predictions_class)
            predictions_class, out_boxes,predictions_mask=\
                self.dn_post_process(predictions_class,out_boxes,mask_dict,predictions_mask)
            predictions_class,predictions_mask=list(predictions_class),list(predictions_mask)
        elif self.training:  # this is to insure self.label_enc participate in the model
            predictions_class[-1] += 0.0*self.label_enc.weight.sum()

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'pred_boxes':out_boxes[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask,out_boxes
            )
        }
        if self.two_stage:
            out['interm_outputs'] = interm_outputs
        out['scda_payload'] = scda_payload
        return out, mask_dict
