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


__all__ = ["VQADatasetMapper"]


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
class VQADatasetMapper:
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
        num_classes=None,
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
        self.num_classes = num_classes

    @classmethod
    def from_config(cls, cfg, is_train=True, dataset_name=None):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        tokenizer = build_tokenizer(cfg['MODEL']['TEXT'])
        max_token_num = 2048
        device = cfg['device']

        ret = {
            "is_train": is_train,
            "dataset_name": dataset_name,
            "tfm_gens": tfm_gens,
            "image_format": cfg['INPUT']['FORMAT'],
            "tokenizer": tokenizer,
            "max_token_num": max_token_num,
            "num_classes": cfg['VQA']['INPUT']['NUM_CLASSES'],
            "device": device,
        }
        return ret

    def get_image(self, inp):
        image_bytes = io.BytesIO(inp)
        image_bytes.seek(0)
        return Image.open(image_bytes)
    
    def preprocess_multimodal(self,
        sources: Sequence[str]):

        for i, sentence in enumerate(sources):
            sources[i] = DEFAULT_IMAGE_TOKEN + '\n' + sentence
            sources[i] = sources[i].strip()

        return sources
    
    def preprocess_v1(self,
        sources,
        labels,
        splits,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False
    ) -> Dict:
        conv = conversation_lib.conv_vicuna_v1.copy() # which version to use?

        # Apply prompt templates
        conversations = []
        conv.messages = []
        for i, sentence in enumerate(sources):
            prompt = "\nAnswer the question using a single word or phrase."

            # Add prompt for VQA
            if i == 0:
                sentence += prompt

            role = conv.roles[0]
            conv.append_message(role, sentence)

            if splits == 'train':
                role = conv.roles[1]
                label_sentence = ', '.join(labels[i])
                conv.append_message(role, label_sentence)
            else:
                role = conv.roles[1]
                label_sentence = ''
                conv.append_message(role, label_sentence)
                        
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

                if i == 0:
                    cur_len += round_len
                else:
                    cur_len += round_len - 1
                    
            target[cur_len:] = IGNORE_INDEX
            if  splits == 'train':
                if cur_len < self.max_token_num:
                    if cur_len != total_len:
                        target[:] = IGNORE_INDEX
                        print(
                            f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                            f" (ignored)"
                        )

        attention_masks = (target != -100).type(torch.float32)


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
        labels = dataset_dict['labels']
        splits = dataset_dict['split']

        sources = self.preprocess_multimodal(copy.deepcopy(sources))
        tokens = self.preprocess_v1(sources, labels, splits, self.tokenizer, has_image=True)
        
        dataset_dict['tokens'] = {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"], "labels": tokens["labels"]}
        return dataset_dict