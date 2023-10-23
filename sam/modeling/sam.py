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
    @torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        
        input_images = torch.stack([self.lbk_preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings, x_list = self.image_encoder(input_images)

        hyper_in_list = []
        upscaled_embedding_list = []
        src_list = []
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
          low_res_masks, iou_predictions, hyper_in, upscaled_embedding, src = self.mask_decoder(
              image_embeddings=curr_embedding.unsqueeze(0),
              image_pe=self.prompt_encoder.get_dense_pe(),
              sparse_prompt_embeddings=sparse_embeddings,
              dense_prompt_embeddings=dense_embeddings,
              multimask_output=multimask_output,
          )
          hyper_in_list.append(hyper_in)
          upscaled_embedding_list.append(upscaled_embedding)
          src_list.append(src)

        return x_list, hyper_in_list, upscaled_embedding_list, src_list
    
    # by lbk edit
    def lbk_preprocess(self, x: torch.Tensor) -> torch.Tensor:
      """Normalize pixel values and pad to a square input."""
      # Normalize colors
      x = (x - self.pixel_mean) / self.pixel_std
      return x
