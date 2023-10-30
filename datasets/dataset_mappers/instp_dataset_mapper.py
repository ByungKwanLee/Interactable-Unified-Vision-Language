# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging

import io
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
import transformers

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog
from typing import Dict, Optional, Sequence, List

from modeling.language.LangEncoder import build_tokenizer
from modeling.utils import configurable
from modeling.language.LangEncoder.constant import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from modeling.language.LangEncoder import conversation as conversation_lib
from modeling.language.LangEncoder.mm_utils import tokenizer_image_token


__all__ = ["InstPreDatasetMapper"]


def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    # The scope of vlp dataset may not need any augmentation.
    cfg_input = cfg['INPUT']
    image_size = cfg_input['IMAGE_SIZE']
    augmentation = []

    augmentation.extend([
        T.Resize((image_size, image_size)),
    ])
    
    return augmentation

# This is specifically designed for the COCO dataset.
class InstPreDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        dataset_name=None,
        *,
        tfm_gens,
        image_format,
        tokenizer=None,
        max_token_num=None,
        device=None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            crop_gen: crop augmentation
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[PretrainDatasetMapper] Full TransformGens used in training: {}".format(
                str(self.tfm_gens)
            )
        )

        self.img_format = image_format
        self.is_train = is_train

        self.all_arrows = MetadataCatalog.get(dataset_name).arrows

        self.tokenizer = tokenizer
        self.max_token_num = max_token_num
        self.device = device

    @classmethod
    def from_config(cls, cfg, is_train=True, dataset_name=None):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        tokenizer = build_tokenizer(cfg['MODEL']['TEXT'])
        max_token_num = cfg['MODEL']['TEXT']['CONTEXT_LENGTH']
        device = cfg['device']

        ret = {
            "is_train": is_train,
            "dataset_name": dataset_name,
            "tfm_gens": tfm_gens,
            "image_format": cfg['INPUT']['FORMAT'],
            "tokenizer": tokenizer,
            "max_token_num": max_token_num,
            "device": device,
        }
        return ret

    def get_image(self, inp):
        image_bytes = io.BytesIO(inp)
        image_bytes.seek(0)
        return Image.open(image_bytes)

    def _add_speaker_and_signal(self, header, source, get_conversation=True):
        """Add speaker and start/end signal on each round."""
        BEGIN_SIGNAL = "### "
        END_SIGNAL = "\n"
        conversation = header
        for sentence in source:
            from_str = sentence["from"]
            if from_str.lower() == "human":
                from_str = conversation_lib.default_conversation.roles[0]
            elif from_str.lower() == "gpt":
                from_str = conversation_lib.default_conversation.roles[1]
            else:
                from_str = 'unknown'
            sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                                sentence["value"] + END_SIGNAL)
            if get_conversation:
                conversation += sentence["value"]
        conversation += BEGIN_SIGNAL
        return conversation

    def preprocess_multimodal(self,
        sources: Sequence[str]):

        for source in sources:
            for sentence in source:
                if DEFAULT_IMAGE_TOKEN in sentence['value']:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                    sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                    sentence['value'] = sentence['value'].strip()

                replace_token = DEFAULT_IMAGE_TOKEN

                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

        return sources

    def preprocess_sys(self,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
    ) -> Dict:
        conv = conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # Apply prompt templates
        conversations = []
        for i, source in enumerate(sources):
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

        # Tokenize conversations

        if has_image:
            input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, self.max_token_num, return_tensors='pt') for prompt in conversations], dim=0)
        else:
            input_ids = tokenizer(
                conversations,
                return_tensors="pt",
                padding="longest",
                max_length=self.max_token_num,
                truncation=True,
            ).input_ids

        targets = input_ids.clone()

        assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

        # Mask targets
        sep = conv.sep + conv.roles[1] + ": "
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep

                if has_image:
                    round_len = len(tokenizer_image_token(rou, tokenizer, self.max_token_num))
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer, self.max_token_num)) - 2
                else:
                    round_len = len(tokenizer(rou).input_ids)
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX

            # if cur_len < self.max_token_num:
            #     if cur_len != total_len:
            #         target[:] = IGNORE_INDEX
            #         print(
            #             f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
            #             f" (ignored)"
            #         )

        attention_masks = (target != -100).type(torch.float32)

        # Pad to length 256
        max_length = 256
        input_ids = F.pad(input_ids, (0, max_length - input_ids.size(-1)))
        targets = F.pad(targets, (0, max_length - targets.size(-1)), value=IGNORE_INDEX)
        attention_masks = F.pad(attention_masks, (0, max_length - attention_masks.size(-1)))

        return dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=attention_masks,
        )

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        arr = self.all_arrows[dataset_dict['arr_id']]
        cur_id = dataset_dict['cur_id']
        image = self.get_image(arr['image'][cur_id].as_py())

        image = utils._apply_exif_orientation(image)
        image = utils.convert_PIL_to_numpy(image, self.img_format)
        utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        sources = dataset_dict['captions']

        sources = self.preprocess_multimodal(copy.deepcopy(sources))

        tokens = self.preprocess_sys(sources, self.tokenizer, has_image=True)

        # tokens = self.tokenizer(
        #     captions, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
        # )
        dataset_dict['tokens'] = {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"].unsqueeze(0), "labels": tokens["labels"]}
        return dataset_dict