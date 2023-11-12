# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import torch
from torch import nn
from torch.nn import functional as F

from timm.models.layers import trunc_normal_

from .build import register_model
from ..utils import configurable
from .LangEncoder import build_tokenizer, build_lang_encoder
from utils.prompt_engineering import prompt_engineering, get_prompt_templates
from modeling.language.LangEncoder.constant import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


class LanguageEncoder(nn.Module):

    @configurable
    def __init__(
        self,
        tokenizer,
        tokenizer_type,
        lang_encoder,
        lang_projection,
        max_token_num,
        queue_operator,
    ):
        super().__init__()
        # seg
        self.tokenizer = tokenizer
        self.tokenizer_type = tokenizer_type
        self.lang_encoder = lang_encoder
        self.lang_proj = lang_projection
        self.max_token_num = max_token_num
        self.logit_scale = nn.Parameter(torch.ones([]))
        
        # captioning & retrieval
        for key, value in queue_operator.items():
            self.register_buffer(key, value)
    
    @property
    def device(self):
        return self.logit_scale.device

    @classmethod
    def from_config(cls, cfg):
        # build up text encoder for seg
        tokenizer = build_tokenizer(cfg['MODEL']['TEXT'])
        tokenizer_type = cfg['MODEL']['TEXT']['TOKENIZER']
        lang_encoder = build_lang_encoder(cfg['MODEL']['TEXT'], tokenizer, cfg['VERBOSE'])
        
        dim_lang = cfg['MODEL']['TEXT']['WIDTH']
        dim_projection = cfg['SYSLEARNER_DIM']
        lang_projection = nn.Parameter(torch.empty(dim_lang, dim_projection))
        trunc_normal_(lang_projection, std=.02)

        # tested not working better      
        queue_operator = {}

        return {
            "tokenizer": tokenizer,
            "tokenizer_type": tokenizer_type,
            "lang_encoder": lang_encoder,
            "lang_projection": lang_projection,
            "max_token_num": 77,
            "queue_operator": queue_operator,
        }

    def get_text_embeddings(self, class_names, name='default', is_eval=False, add_bgd=False, prompt=True, norm=True, store_buffer=None):
        if not is_eval:
            if prompt:
                # randomly sample one template
                arbitary_concepts = [
                    prompt_engineering(class_names[label].replace('-other','').replace('-merged','').replace('-stuff',''), topk=10000, suffix='.') \
                    for label in range(len(class_names))
                ]
                if add_bgd:
                    arbitary_concepts.append("A background in coco.")
            else:
                arbitary_concepts = class_names
            
            input_ids = []
            attention_masks = []
            for txt in arbitary_concepts:
                tokens = self.tokenizer(
                    txt, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
                )
                tokens['input_ids'].squeeze_()
                tokens['attention_mask'].squeeze_()

                input_ids.append(tokens['input_ids'])
                attention_masks.append(tokens['attention_mask'])

            arbitary_tokens = torch.stack(input_ids)
            arbitary_attention_masks = torch.stack(attention_masks)

            text_emb = self.forward_language((arbitary_tokens.cuda(), arbitary_attention_masks.cuda()), norm=norm)
            setattr(self, '{}_text_embeddings'.format(name), text_emb)
        else:
            with torch.no_grad():
                def extract_mean_emb(txts):
                    tokens = self.tokenizer(
                        txts, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
                    )
                    clss_embedding = self.forward_language((tokens['input_ids'].cuda(), tokens['attention_mask'].cuda()), norm=norm)
                    clss_embedding = clss_embedding.mean(dim=0)
                    clss_embedding /= clss_embedding.norm()
                    return clss_embedding

                templates = get_prompt_templates()
                clss_embeddings = []
                if prompt:
                    for clss in class_names:
                        txts = [template.format(clss.replace('-other','').replace('-merged','').replace('-stuff','')) for template in templates]
                        clss_embeddings.append(extract_mean_emb(txts))
                else:
                    for clss in class_names:
                        clss_embeddings.append(extract_mean_emb([clss]))

                if add_bgd:
                    txts = ["A background in coco."]
                    clss_embeddings.append(extract_mean_emb(txts))

                text_emb = torch.stack(clss_embeddings, dim=0)
                setattr(self, '{}_text_embeddings'.format(name), text_emb)

    def reset_text_embeddings(self, name='default'):
        pass

    def get_text_token_embeddings(self, txts, name='default', token=False, norm=False):
        if not token:
            tokens = self.tokenizer(
                txts, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
            )
            tokens = {key: value.cuda() for key, value in tokens.items()}
        else:
            tokens = txts
        token_emb, class_emb = self.forward_language_token((tokens['input_ids'], tokens['attention_mask']), norm=norm)
        ret = {"tokens": tokens,
                "token_emb": token_emb,
                "class_emb": class_emb,}
        setattr(self, '{}_token_embeddings'.format(name), ret)
        return ret
    
    def get_instruction_token_embeddings(self, tokens, name='default', token=False, norm=False):
        max_len = self.max_token_num
        new_input_embeds = []
        new_labels = [] if tokens['labels'] is not None else None
        new_input_embeds_align = []
        new_labels_align = []

        for batch_idx, cur_input_ids in enumerate(tokens['input_ids']):
            cur_new_labels = []
            
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            image_token_start = image_token_indices[0]
                        
            new_input_embeds.append(cur_input_ids[cur_input_ids != -200])
            cur_labels = tokens['labels'][batch_idx]
            
            cur_new_labels.append(cur_labels[:image_token_start])
            cur_new_labels.append(cur_labels[image_token_start + 1:])

            cur_new_labels = torch.cat(cur_new_labels, dim=0)
            new_labels.append(cur_new_labels)

        for cur_new_embed in new_input_embeds:
            cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
            new_input_embeds_align.append(cur_new_embed)
        new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

        for cur_new_label in new_labels:
            cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
            new_labels_align.append(cur_new_label)
        new_labels = torch.stack(new_labels_align, dim=0)

        tokens['input_ids'] = new_input_embeds
        tokens['labels'] = new_labels
        
        new_attention_mask = (new_labels != -100).int()
        tokens['attention_mask'] = new_attention_mask
        
        class_input_embeds_ = new_input_embeds * new_attention_mask.type(torch.int16) 
        
        token_emb, _ = self.forward_language_token((new_input_embeds, new_attention_mask), norm=norm)
        class_input_embeds = [x[x.nonzero()[0].item():] for x in class_input_embeds_]

        new_class_embeds_align = []
        for cur_new_embed in class_input_embeds:
            cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
            new_class_embeds_align.append(cur_new_embed)
        new_class_embeds = torch.stack(new_class_embeds_align, dim=0)

        _, class_emb = self.forward_language_token((new_class_embeds, new_attention_mask), norm=norm)
        ret = {"tokens": tokens,
                "token_emb": token_emb,
                "class_emb": class_emb,}
        setattr(self, '{}_token_embeddings'.format(name), ret)
        return ret


    def forward_language(self, texts, norm=True):
        x = self.lang_encoder(*texts)
        x = x['last_hidden_state']

        if self.tokenizer_type == 'clip':
            x = x[torch.arange(x.size(0)), texts[0].argmax(dim=-1)]
        else:
            x = x[:, 0]

        x = x @ self.lang_proj
        if norm:
            x = x / (x.norm(dim=-1, keepdim=True) + 1e-7)
        return x
    
    def forward_language_token(self, texts, norm=False):
        x = self.lang_encoder(*texts)
        token_x = x['last_hidden_state']

        if self.tokenizer_type == 'clip':
            class_x = token_x[torch.arange(token_x.size(0)), texts[0].argmax(dim=-1)]
        else:
            class_x = token_x[:, 0]

        class_x = class_x @ self.lang_proj
        token_x = token_x @ self.lang_proj

        if norm:
            class_x = class_x / (class_x.norm(dim=-1, keepdim=True) + 1e-7)
            token_x = token_x / (token_x.norm(dim=-1, keepdim=True) + 1e-7)

        return token_x, class_x
    
    def compute_similarity(self, v_emb, name='default', fake=False):
        if fake:
            return None
        v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)
        t_emb = getattr(self, '{}_text_embeddings'.format(name))
        output = self.logit_scale.exp() * v_emb @ t_emb.unsqueeze(0).transpose(1, 2)
        return output


@register_model
def get_language_model(cfg, **kwargs):
    return LanguageEncoder(cfg)