# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from typing import Any, Dict, List

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder


class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    # Batch Individual Mask Generation by LBK
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool = False,
    ) -> List[Dict[str, torch.Tensor]]:
        
        input_images = torch.stack([x["image"] for x in batched_input], dim=0)
        image_embeddings, hier_embeddings_dict = self.image_encoder(input_images)
        
        # prompting encodeing for flexible inputs
        _, _, H, W = input_images.shape
        self.prompt_encoder.input_image_size = (H, W)
        self.prompt_encoder.image_embedding_size = (H//16, W//16)

        # LBK Pop
        src_list = []
        hyper_in_list = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
          if "point_coords" in image_record:
              points = (image_record["point_coords"], image_record["point_labels"])
          else:
              points = None
          sparse_embeddings, dense_embeddings = self.prompt_encoder(
              points=points,
              boxes=image_record.get("boxes", None),
              masks=image_record.get("mask_inputs", None),
          )
          src_outputs, hyper_in = self.mask_decoder(
              image_embeddings=curr_embedding.unsqueeze(0),
              image_pe=self.prompt_encoder.get_dense_pe(),
              sparse_prompt_embeddings=sparse_embeddings,
              dense_prompt_embeddings=dense_embeddings,
              multimask_output=multimask_output,
          )
          src_list.append(src_outputs)
          hyper_in_list.append(hyper_in[:, 0, :])

        # output format transformation, LBK EDIT
        src_output_features = torch.cat([x.unsqueeze(0) for x in src_list], dim=0)
        hyper_in_features = torch.cat([x.unsqueeze(0) for x in hyper_in_list], dim=0)

        return hier_embeddings_dict, src_output_features, hyper_in_features


    # Image Embedding for Interactive SAM
    def forward_image_embedding(self, images):
        image_embeddings, hier_embeddings_dict = self.image_encoder(images)
        return image_embeddings, hier_embeddings_dict

    # Image Embedding for Interactive SAM
    def decode_from_embedding(
        self,
        image_embeddings: torch.Tensor,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool=False,
    ) -> List[Dict[str, torch.Tensor]]:
        
        src_list = []
        hyper_in_list = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
          if "point_coords" in image_record:
              points = (image_record["point_coords"], image_record["point_labels"])
          else:
              points = None
          sparse_embeddings, dense_embeddings = self.prompt_encoder(
              points=points,
              boxes=image_record.get("boxes", None),
              masks=image_record.get("mask_inputs", None),
          )
          src_outputs, hyper_in = self.mask_decoder(
              image_embeddings=curr_embedding.unsqueeze(0),
              image_pe=self.prompt_encoder.get_dense_pe(),
              sparse_prompt_embeddings=sparse_embeddings,
              dense_prompt_embeddings=dense_embeddings,
              multimask_output=multimask_output,
          )
          src_list.append(src_outputs)
          hyper_in_list.append(hyper_in[:, 0, :])

        # output format transformation, LBK EDIT
        src_output_features = torch.cat([x.unsqueeze(0) for x in src_list], dim=0)
        hyper_in_features = torch.cat([x.unsqueeze(0) for x in hyper_in_list], dim=0)

        return src_output_features, hyper_in_features