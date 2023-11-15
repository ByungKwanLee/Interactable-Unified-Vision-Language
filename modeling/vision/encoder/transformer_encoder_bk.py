# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Callable, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

import fvcore.nn.weight_init as weight_init
from detectron2.layers import Conv2d, get_norm

from .build import register_encoder
from ...utils import configurable


# This is a modified FPN decoder.
class BasePixelDecoder(nn.Module):
    def __init__(
        self,
        sam_size: str,
        conv_dim: int,
        mask_dim: int,
        mask_on: bool,
        norm: Optional[Union[str, Callable]] = None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()

        self.in_features = ['res5']  # only 'res5'
        if sam_size == 'base':
            feature_channels = [768] # LBK EDIT
        elif sam_size =='large':
            feature_channels = [1024] # LBK EDIT
        elif sam_size =='huge':
            feature_channels = [1280] # LBK EDIT

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                output_norm = get_norm(norm, conv_dim)
                output_conv = Conv2d(
                    in_channels,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(None)
                output_convs.append(output_conv)
            else:
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)

                lateral_conv = Conv2d(
                    in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
                )
                output_conv = Conv2d(
                    conv_dim,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                weight_init.c2_xavier_fill(lateral_conv)
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("adapter_{}".format(idx + 1), lateral_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        self.mask_on = mask_on
        if self.mask_on:
            self.mask_dim = mask_dim
            self.mask_features = Conv2d(
                conv_dim,
                mask_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            weight_init.c2_xavier_fill(self.mask_features)

        self.maskformer_num_feature_levels = 3  # always use 3 scales

    @classmethod
    def from_config(cls, cfg):
        enc_cfg = cfg['MODEL']['ENCODER']
        ret = {}
        ret["sam_size"] = cfg['SAM_SIZE']
        ret["conv_dim"] = cfg['SYSLEARNER_DIM']
        ret["mask_dim"] = cfg['SYSLEARNER_DIM']
        ret["norm"] = enc_cfg['NORM']
        return ret

    def forward(self, features):
        multi_scale_features = []
        num_cur_levels = 0
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if lateral_conv is None:
                y = output_conv(x)
            else:
                cur_fpn = lateral_conv(x)
                # Following FPN implementation, we use nearest upsampling here
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(y)
                num_cur_levels += 1
        
        mask_features = self.mask_features(y) if self.mask_on else None
        return mask_features, multi_scale_features




# This is a modified FPN decoder with extra Transformer encoder that processes the lowest-resolution feature map.
class TransformerEncoderPixelDecoder(BasePixelDecoder):
    @configurable
    def __init__(
        self,
        sam_size: str,
        conv_dim: int,
        mask_dim: int,
        mask_on: int,
        norm: Optional[Union[str, Callable]] = None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__(sam_size=sam_size, conv_dim=conv_dim, mask_dim=mask_dim, norm=norm, mask_on=mask_on)

        self.in_features = ['res5']  # starting from "res2" to "res5"

        self.seg_head = nn.Sequential(Conv2d(256, 256, kernel_size=1),
                                      nn.Dropout(),
                                      Conv2d(256, 256, kernel_size=1),
                                      nn.ReLU(),
                                      )
        self.input_proj = Conv2d(256, conv_dim, kernel_size=1)


        # update layer
        use_bias = norm == ""
        output_norm = get_norm(norm, conv_dim)
        output_conv = Conv2d(
            conv_dim,
            conv_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            norm=output_norm,
            activation=F.relu,
        )
        weight_init.c2_xavier_fill(output_conv)
        delattr(self, "layer_{}".format(len(self.in_features)))
        self.add_module("layer_{}".format(len(self.in_features)), output_conv)
        self.output_convs[0] = output_conv

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret['mask_on'] = cfg['MODEL']['DECODER']['MASK']
        return ret

    def forward(self, features, src_list):
        multi_scale_features = []
        
        # avg prompt tensor
        src_avg_tensor = torch.cat([src.unsqueeze(0) for src in src_list], dim=0).mean(dim=1)
        
        # Reverse feature maps into top-down order (from low to high resolution)
        output_conv = self.output_convs[0]
        y = output_conv(self.input_proj(self.seg_head(src_avg_tensor)))
        multi_scale_features.append(y)
        mask_features = self.mask_features(y) if self.mask_on else None
        return mask_features, multi_scale_features

@register_encoder
def get_transformer_encoder_fpn(cfg):
    """
    Build a pixel decoder from `cfg.MODEL.MASK_FORMER.PIXEL_DECODER_NAME`.
    """
    return TransformerEncoderPixelDecoder(cfg)