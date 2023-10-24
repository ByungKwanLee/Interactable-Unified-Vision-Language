# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict

from torch import nn

from .registry import register_body
from ..vision.encoder import build_encoder
from ..interface import build_decoder
from ..utils import configurable


class XdecoderHead(nn.Module):

    @configurable
    def __init__(
        self,
        num_classes: int,
        pixel_decoder: nn.Module,
        transformer_predictor: nn.Module,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
        """
        super().__init__()
        self.num_classes = num_classes
        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor

    @classmethod
    def from_config(cls, cfg, lang_encoder: nn.Module, extra: dict):
        enc_cfg = cfg['MODEL']['ENCODER']
        return {
            "num_classes": enc_cfg.get('NUM_CLASSES', None),
            "pixel_decoder": build_encoder(cfg),
            "transformer_predictor": build_decoder(
                cfg,
                lang_encoder,
                mask_classification=True,
                extra=extra,
            ),
        }

    def forward(self, features, upscaled_embedding_list, src_list, mask=None, target_queries=None, target_vlp=None, task='seg', extra={}):
        mask_features, multi_scale_features = self.pixel_decoder(features, src_list)
        predictions = self.predictor(upscaled_embedding_list, multi_scale_features, mask_features, mask, target_queries, target_vlp, task, extra)
        return predictions

@register_body
def get_xdecoder_head(cfg, lang_encoder, extra):
    return XdecoderHead(cfg, lang_encoder, extra)