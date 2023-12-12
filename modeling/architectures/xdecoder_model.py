# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu), Ziyi Dou, Jianwei Yang
# --------------------------------------------------------

from typing import Tuple
import random

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from kornia.contrib import distance_transform

from timm.models.layers import trunc_normal_
from nltk.stem.lancaster import LancasterStemmer
from detectron2.structures import Boxes, ImageList, Instances, BitMasks, BoxMode
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.data import MetadataCatalog

from .build import register_model
from ..utils import configurable, get_class_names, Spatial_ImageList, get_iou
from ..body import build_xdecoder_head
from ..modules import sem_seg_postprocess, SetCriterion, HungarianMatcher, bbox_postprocess
from ..language import build_language_encoder
from ..language.loss import vl_similarity, image_text_contrastive_loss_queue
from utils.prompt_engineering import prompt_engineering
from utils.constants import COCO_PANOPTIC_CLASSES
from llm.load_llm import prepare_llm
from sam.utils.amg import build_all_layer_point_grids
from sam import build_sam


st = LancasterStemmer()


class GeneralizedXdecoder(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        syslearner_dim: int,
        sam_size:str,
        load_llm: bool,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        losses: dict,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        task_switch: dict,
        phrase_prob: float,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        train_dataset_name: str,
        retrieval_emsemble: bool,
        dim_proj: int,
    ):

        super().__init__()
        self.sam_size = sam_size
        if sam_size=='base':
            sam = build_sam.sam_model_registry['vit_b'](checkpoint='sam/ckpt/sam_vit_b_01ec64.pth')
        elif sam_size=='large':
            sam = build_sam.sam_model_registry['vit_l'](checkpoint='sam/ckpt/sam_vit_l_0b3195.pth')
        elif sam_size=='huge':
            sam = build_sam.sam_model_registry['vit_h'](checkpoint='sam/ckpt/sam_vit_h_4b8939.pth')

        # LBK build LLM
        if load_llm:
            bits = 8
            self.bit_flag = True if bits in [4, 8] else False
            self.llm, self.llm_tokenizer, self.data_collator = prepare_llm(bits=bits) #bits=4
            self.img_to_lang = nn.Linear(syslearner_dim, 4096)
                
        self.sem_seg_head = sem_seg_head
        self.backbone = sam.image_encoder # sam
        self.criterion = criterion
        self.losses = losses
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata

        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on

        # caption argument
        self.task_switch = task_switch
        self.phrase_prob = phrase_prob

        self.test_topk_per_image = test_topk_per_image
        self.train_class_names = get_class_names(train_dataset_name)

        self.retrieval_emsemble = retrieval_emsemble
        # backbone itc loss
        if task_switch['retrieval'] and retrieval_emsemble:
            self.backbone_proj = nn.Parameter(torch.empty(768, dim_proj))
            trunc_normal_(self.backbone_proj, std=.02)

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

    @classmethod
    def from_config(cls, cfg):
        dec_cfg = cfg['MODEL']['DECODER']

        # Loss parameters:
        no_object_weight = dec_cfg['NO_OBJECT_WEIGHT']

        # loss weights, switcher for task, and top layers to compute loss
        loss_weights = {'mask': {'ce': dec_cfg['CLASS_WEIGHT'], 'dice': dec_cfg['DICE_WEIGHT'], 'bce': dec_cfg['MASK_WEIGHT']},
                        'bbox': {'l1': dec_cfg['BBOX_WEIGHT'], 'giou': dec_cfg['GIOU_WEIGHT']},
                        'caption': dec_cfg['CAPTION_WEIGHT'],
                        'captioning': dec_cfg['CAPTIONING_WEIGHT'], 
                        'retrieval': {'decoder': dec_cfg['RETRIEVAL_WEIGHT'], 'backbone': dec_cfg['BACKBONER_WEIGHT']},
                        'grounding': {'ce': dec_cfg['GCLASS_WEIGHT'], 'dice': dec_cfg['GDICE_WEIGHT'], 'bce': dec_cfg['GMASK_WEIGHT']}}

        task_switch = {'bbox': dec_cfg.get('DETECTION', False),
                       'mask': dec_cfg.get('MASK', True),
                       'caption': dec_cfg['CAPTION'].get('ENABLED', False),
                       'captioning': dec_cfg['CAPTIONING'].get('ENABLED', False),
                       'retrieval': dec_cfg['RETRIEVAL'].get('ENABLED', False),
                       'grounding': dec_cfg['GROUNDING'].get('ENABLED', False)}

        top_x_layers = {'mask': dec_cfg.get('TOP_MASK_LAYERS', 10),
                        'caption': dec_cfg.get('TOP_CAPTION_LAYERS', 10), 
                        'captioning': dec_cfg.get('TOP_CAPTIONING_LAYERS', 10),
                        'retrieval': dec_cfg.get('TOP_RETRIEVAL_LAYERS', 10),
                        'grounding': dec_cfg.get('TOP_GROUNDING_LAYERS', 10),}

        # build model
        extra = {'task_switch': task_switch}
        lang_encoder = build_language_encoder(cfg)        
        sem_seg_head = build_xdecoder_head(cfg, lang_encoder, extra)

        # building criterion
        matcher = HungarianMatcher(
            cost_class=loss_weights['mask']['ce'],
            cost_mask=loss_weights['mask']['bce'],
            cost_dice=loss_weights['mask']['dice'],
            num_points=dec_cfg['TRAIN_NUM_POINTS'],
        )

        # init weight dict and criterion loss functions.
        losses = {'seg': [], 'vlp': []}
        if task_switch['mask']:
            losses['seg'] += ["labels", "masks"]
        if task_switch['caption']:
            losses['seg'] += ["captions"]
        if task_switch['grounding']:
            losses['seg'] += ["groundings"]
        if task_switch['captioning']:
            losses['vlp'] += ["captionings"]
        if task_switch['retrieval']:
            losses['vlp'] += ["retrievals"]

        weight_dict = {}
        for key, turn_on in task_switch.items():
            if turn_on:
                if isinstance(loss_weights[key], dict):
                    # HACK it should support bbox in the future
                    for key_, weight in loss_weights[key].items():
                        weight_dict["loss_{}_{}_0".format(key, key_)] = weight # NOTE: hard code for segmentation that has multiple loss
                else:
                    weight_dict["loss_{}_0".format(key)] = loss_weights[key]
        
        # generate full weight dict and remove not computed layers. 
        aux_weight_dict = {}
        for i in range(9):
            for k, v in weight_dict.items():
                if (i+1) > (top_x_layers[k.split('_')[1]] - 1):
                    continue
                aux_weight_dict.update({k.replace('_0', f"_{i+1}"): v})
        weight_dict.update(aux_weight_dict)
        
        
        # llm loss, LBK EDIT
        weight_dict.update({'loss_llm': 1.0})

        grd_weight = {'text': dec_cfg['GROUNDING']['TEXT_WEIGHT'], 'class': dec_cfg['GROUNDING']['CLASS_WEIGHT']}
        # generate critenrion for loss function.
        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            top_x_layers=top_x_layers,
            eos_coef=no_object_weight,
            losses=[],
            num_points=dec_cfg['TRAIN_NUM_POINTS'],
            oversample_ratio=dec_cfg['OVERSAMPLE_RATIO'],
            importance_sample_ratio=dec_cfg['IMPORTANCE_SAMPLE_RATIO'],
            grounding_weight=grd_weight,
        )

        # extra logistic
        train_dataset_name = cfg['DATASETS']['TRAIN'][0] # HACK for only one training set.
        phrase_prob = dec_cfg['CAPTION'].get('PHRASE_PROB', 0.5)

        return {
            "syslearner_dim": cfg['SYSLEARNER_DIM'],
            "sam_size": cfg['SAM_SIZE'],
            "load_llm": cfg['Load_LLM'],
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "losses": losses,
            "num_queries": cfg['MASK_PROPOSAL']+1,
            "object_mask_threshold": dec_cfg['TEST']['OBJECT_MASK_THRESHOLD'],
            "overlap_threshold": dec_cfg['TEST']['OVERLAP_THRESHOLD'],
            "metadata": MetadataCatalog.get(cfg['DATASETS']['TRAIN'][0]),
            "sem_seg_postprocess_before_inference": (
                dec_cfg['TEST']['SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE']
                or dec_cfg['TEST']['PANOPTIC_ON']
                or dec_cfg['TEST']['INSTANCE_ON']
            ),
            "pixel_mean": cfg['INPUT']['PIXEL_MEAN'],
            "pixel_std": cfg['INPUT']['PIXEL_STD'],
            "task_switch": task_switch,
            "phrase_prob": phrase_prob,
            # inference
            "semantic_on": dec_cfg['TEST']['SEMANTIC_ON'],
            "instance_on": dec_cfg['TEST']['INSTANCE_ON'],
            "panoptic_on": dec_cfg['TEST']['PANOPTIC_ON'],
            "test_topk_per_image": cfg['COCO']['TEST']['DETECTIONS_PER_IMAGE'],
            "train_dataset_name": train_dataset_name,
            "retrieval_emsemble": dec_cfg['RETRIEVAL']['ENSEMBLE'],
            "dim_proj": cfg['SYSLEARNER_DIM'],
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, mode=None):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        # visualization
        # a = batched_inputs['coco'][1]['image'].permute(1,2,0).flip(2).cpu().numpy()
        # b = batched_inputs['coco'][0]['instances']._fields['gt_masks'][0].cpu().numpy()
        # c = batched_inputs['vlp'][0]['image'].permute(1,2,0).flip(2).cpu().numpy()
        
        if self.training:
            losses = {}
            if self.task_switch['mask'] and (not 'instruction' in batched_inputs.keys()):
                if 'coco' in batched_inputs.keys(): losses.update(self.forward_seg(batched_inputs['coco']))
            if self.task_switch['retrieval'] or self.task_switch['captioning']:
                                
                if 'vlp' in batched_inputs.keys():
                    losses.update(self.forward_vlp(batched_inputs['vlp']))
                elif 'instp' in batched_inputs.keys():
                    losses.update(self.forward_vlp(batched_inputs['instp']))
                elif 'instruction' in batched_inputs.keys():
                    losses.update(self.forward_llm(batched_inputs['instruction']))

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    # if  ('grounding' in k) or ('mask' in k): # fine-tuning for ref-coco, LBK EDIT
                    if True:
                        losses[k] *= self.criterion.weight_dict[k]
                    else: # remove this loss if not specified in `weight_dict`
                        losses[k] *= 0
                else: # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            if mode == 'llm_captioning':
                results = self.evaluate_llm_captioning(batched_inputs)
            elif mode == 'grounding_refcoco':
                results = self.evaluate_grounding(batched_inputs)
            elif mode == 'interactive':
                results = self.evaluate_interactive(batched_inputs)
            elif mode == 'vqa':
                results = self.evaluate_vqa(batched_inputs)
            else:
                results = self.evaluate(batched_inputs)
            return results
        
    # LBK SAM Input Generator
    # def sam_input_generator(self, images):
    #     # input_point = torch.as_tensor(build_all_layer_point_grids(self.num_grids_horizon, 0, 1)[0] * images.shape[2], dtype=torch.int64).cuda()
    #     # input_label = torch.tensor([1 for _ in range(input_point.shape[0])]).cuda()
    #     sam_input = [
    #         {
    #             'image': i,
    #             # 'point_coords': input_point,
    #             # 'point_labels': input_label,
    #         } for i in images
    #     ]
    #     return sam_input


    def forward_seg(self, batched_inputs):
        images = [x["image"].to(self.device).flip(0) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, 32)

        self.sem_seg_head.predictor.lang_encoder.get_text_embeddings(self.train_class_names, is_eval=False)

        extra = {}
        # mask classification target
        if "instances" in batched_inputs[0]:
            # input bounding box is checked to be correct.
            targets = self.prepare_targets(batched_inputs, images)

            if self.task_switch['grounding']:
                grounding_tokens = [x['grounding_query_embs'] for x in targets] # need to pad for more than one grounding token
                grounding_tokens = nn.utils.rnn.pad_sequence(grounding_tokens)
                extra['grounding_tokens'] = grounding_tokens

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features, extra=extra)

        _outputs = {}
        for key, value in outputs.items():
            if key == 'pred_logits':
                _outputs[key] = value[:,:self.num_queries-1]
            elif key == 'pred_masks':
                _outputs[key] = value[:,:self.num_queries-1]
                if self.task_switch['grounding']:
                    _outputs['pred_gmasks'] = value[:,self.num_queries:2*self.num_queries-1]
            elif key == 'pred_captions':
                _outputs[key] = value[:,:self.num_queries-1]
                if self.task_switch['grounding']:
                    _outputs['pred_gtexts'] = value[:,self.num_queries:2*self.num_queries-1]
            elif key == 'aux_outputs':
                _outputs[key] = []
                for i in range(len(value)):
                    _outputs[key] += [{}]
                    for _key, _value in value[i].items():
                        if _key == 'pred_logits':
                            _outputs[key][i][_key] = _value[:,:self.num_queries-1]
                        elif _key == 'pred_masks':
                            _outputs[key][i][_key] = _value[:,:self.num_queries-1]
                            if self.task_switch['grounding']:
                                _outputs[key][i]['pred_gmasks'] = _value[:,self.num_queries:2*self.num_queries-1]
                        elif _key == 'pred_captions':
                            _outputs[key][i][_key] = _value[:,:self.num_queries-1]
                            if self.task_switch['grounding']:
                                _outputs[key][i]['pred_gtexts'] = _value[:,self.num_queries:2*self.num_queries-1]        
        outputs = _outputs

        extra = {'lang_logit': self.sem_seg_head.predictor.lang_encoder.logit_scale,
                 'class_embeddings': getattr(self.sem_seg_head.predictor.lang_encoder, '{}_text_embeddings'.format('default'))}

        # bipartite matching-based loss
        self.criterion.losses = self.losses['seg'] # seg criterion losses
        losses = self.criterion(outputs, targets, extra)

        del outputs
        del _outputs
        return losses

    def forward_vlp(self, batched_inputs):
        images = torch.cat([x["image"].flip(0).to(self.device).unsqueeze(0) for x in batched_inputs], dim=0)
        images = (images - self.pixel_mean) / self.pixel_std
                
        targets_vlp = self.prepare_vlp_targets(batched_inputs)

        extra = {"token_embedding": self.sem_seg_head.predictor.lang_encoder.lang_encoder.token_embedding,
                 "lang_encoder": self.sem_seg_head.predictor.lang_encoder,
                 "training": self.training}

        # LBK SAM propagation
        hier_embeddings_dict, src_output_features, hyper_in_features = self.backbone(self.sam_input_generator(images))
        outputs = self.sem_seg_head(hier_embeddings_dict, src_output_features, hyper_in_features, target_queries=None, target_vlp=targets_vlp, task='vlp', extra=extra)

        for key, value in outputs.items():
            if key == 'pred_captionings':
                outputs[key] = value
            elif key == 'pred_captions':
                # outputs[key] = value[:,-1:]
                outputs[key] = value
            elif key == 'aux_outputs':
                outputs[key] = []
                for i in range(len(value)):
                    outputs[key] += [{}]
                    for _key, _value in value[i].items():
                        if _key == 'pred_captions':
                            # outputs[key][i][_key] = _value[:,-1:]
                            outputs[key][i][_key] = _value
                        elif _key == 'pred_captionings':
                            outputs[key][i][_key] = _value

        self.criterion.losses = self.losses['vlp'] # seg criterion losses
        losses = self.criterion.forward_vlp(outputs, targets_vlp, extra)
        del outputs

        if self.task_switch['retrieval'] and self.retrieval_emsemble:
            # compute backbone vlp.
            v_emb = hier_embeddings_dict['res5']
            bs,nc,_,_ = v_emb.shape
            v_emb = v_emb.reshape(bs,nc,-1)
            v_emb = F.adaptive_avg_pool1d(v_emb, 1).reshape(bs,nc) @ self.backbone_proj
            t_emb = torch.cat([x['caption_proj'] for x in targets_vlp], dim=0)
            loss_contrast = image_text_contrastive_loss_queue(v_emb, t_emb, self.sem_seg_head.predictor.lang_encoder, None)
            losses['loss_retrieval_backbone_0'] = loss_contrast
        return losses

    def forward_llm(self, batched_inputs):
        # task switch False
        self.task_switch['mask']=False
        
        images = torch.cat([x["image"].flip(0).to(self.device).unsqueeze(0) for x in batched_inputs], dim=0)
        images = (images - self.pixel_mean) / self.pixel_std

        extra = {"token_embedding": self.sem_seg_head.predictor.lang_encoder.lang_encoder.token_embedding,
                 "lang_encoder": self.sem_seg_head.predictor.lang_encoder,
                 "training": self.training}

       # LBK SAM propagation
        outputs = self.sem_seg_head(*self.backbone(self.sam_input_generator(images)), target_queries=None, target_vlp=None, task='llm', extra=extra)
        
        targets_llm = self.prepare_llm_targets(batched_inputs)
        llm_outputs = self.llm(
            input_ids=targets_llm["input_ids"],
            attention_mask=targets_llm["attention_mask"],
            labels=targets_llm["labels"],
            images=self.img_to_lang(outputs['image_feature'][-1].detach()),
            bit_flag=self.bit_flag
        )

        # task switch True
        self.task_switch['mask']=True
        
        return {'loss_llm': llm_outputs.loss}

    def create_pascal_label_colormap(self):
        def bit_get(val, idx):
            return (val >> idx) & 1
        colormap = np.zeros((512, 3), dtype=int)
        ind = np.arange(512, dtype=int)

        for shift in reversed(list(range(8))):
            for channel in range(3):
                colormap[:, channel] |= bit_get(ind, channel) << shift
            ind >>= 3

        return colormap / 255
    

    def evaluate(self, batched_inputs):
        images = [x["image"].to(self.device).flip(0) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        
        images = ImageList.from_tensors(images, 32)
        targets = targets_grounding = queries_grounding = None
        outputs = self.sem_seg_head(self.backbone(images.tensor), target_queries=queries_grounding)

        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]
        box_pred_results = outputs["pred_boxes"] if self.task_switch['bbox'] else [None for i in range(len(mask_pred_results))]

        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        input_size = mask_pred_results.shape[-2:]
        del outputs

        processed_results = []
        for mask_cls_result, mask_pred_result, box_pred_result, input_per_image, image_size in zip(
            mask_cls_results, mask_pred_results, box_pred_results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results.append({})

            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, image_size, height, width
                )
                mask_cls_result = mask_cls_result.to(mask_pred_result)

            # semantic segmentation inference
            if self.semantic_on:
                r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                if not self.sem_seg_postprocess_before_inference:
                    r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                processed_results[-1]["sem_seg"] = r

            # panoptic segmentation inference
            if self.panoptic_on:
                panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                processed_results[-1]["panoptic_seg"] = panoptic_r
            
            # LBK Visualization
            # cmap = self.create_pascal_label_colormap()
            # c = cmap[panoptic_r[0].cpu()]
            # img = input_per_image['image'].flip(0).permute(1,2,0).cpu().numpy()
            # m = (mask_pred_result>0)[0].unsqueeze(2).cpu().numpy()
            # label = cmap[batched_inputs[0]['sem_seg'].cpu()]

            # instance segmentation inference
            if self.instance_on:
                if self.task_switch['bbox']:
                    box_pred_result = bbox_postprocess(box_pred_result, input_size, image_size, height, width)
                instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result, box_pred_result)
                processed_results[-1]["instances"] = instance_r

        return processed_results
    
    
    def evaluate_llm_captioning(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, 32)

        extra = {"token_embedding": self.sem_seg_head.predictor.lang_encoder.lang_encoder.token_embedding,
                 "lang_encoder": self.sem_seg_head.predictor.lang_encoder,
                 "training": self.training}

        if not hasattr(self, 'start_token'):
            self.start_token = torch.tensor([[49406]*77], device=self.device)
        
        targets = targets_grounding = queries_grounding = None

        captioning_mask = None
        if 'captioning_mask' in batched_inputs[-1]:
            captioning_mask = torch.cat([x['captioning_mask'] for x in batched_inputs])


        # LBK SAM propagation
        outputs = self.sem_seg_head(*self.backbone(self.sam_input_generator(images)), target_queries=queries_grounding, task='llm', extra=extra)

        processed_results = []
        for idx, batch_data in enumerate(batched_inputs):
            input_ids = batch_data['tokens']['prompt_ids']

            with torch.inference_mode():
                output_ids = self.llm.generate(
                    input_ids=input_ids,
                    images=self.img_to_lang(outputs['image_feature'][-1]),
                    max_new_tokens=128,
                    min_length=1,
                    num_beams=5)
            
            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            llm_outputs = self.llm_tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            llm_outputs = llm_outputs.strip()

            processed_results.append({"captioning_token": output_ids,
                                    "captioning_text": llm_outputs.split('.')[0],
                                    "image_id": batch_data['image_id']})
        return processed_results
    
    def evaluate_vqa(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, 32)

        extra = {"token_embedding": self.sem_seg_head.predictor.lang_encoder.lang_encoder.token_embedding,
                 "lang_encoder": self.sem_seg_head.predictor.lang_encoder,
                 "training": self.training}
        
        # LBK SAM propagation
        outputs = self.sem_seg_head(*self.backbone(self.sam_input_generator(images)), target_queries=None, task='vqa', target_vlp=None, extra=extra)

        
        processed_results = []
        for i, batch_data in enumerate(batched_inputs):
            idx = batch_data["question_ids"][0]
            cur_prompt = batch_data["captions"][0]
            input_ids = batch_data['tokens']['input_ids']

            with torch.inference_mode():
                output_ids = self.llm.generate(
                    input_ids=input_ids,
                    images=self.img_to_lang(outputs['image_feature'][-1]),
                    max_new_tokens=10,
                    min_length=1,
                    num_beams=5)
            
            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            llm_outputs = self.llm_tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            llm_outputs = llm_outputs.strip()

            processed_results.append({"question_id": idx,
                                    "prompt": cur_prompt,
                                    "text": llm_outputs})
        return processed_results


    def evaluate_grounding(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, 32)
        features = self.backbone(images.tensor)
        extra = {}

        # comment for multi object inference.
        mask_pred_results = []
        for idx, batch_per_image in enumerate(batched_inputs):
            grd_texts = batch_per_image['groundings']['texts']
            grd_texts = [x[0] for x in grd_texts]

            gtext = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(grd_texts, name='grounding', token=False, norm=False)
            token_emb = gtext['token_emb']
            tokens = gtext['tokens']
            query_emb = token_emb[tokens['attention_mask'].bool()]
            extra['grounding_tokens'] = query_emb[:,None]

            outputs = self.sem_seg_head(features, extra=extra, task='grounding_eval')

            pred_gmasks = outputs['pred_masks'][idx,self.num_queries:2*self.num_queries-1]
            v_emb = outputs['pred_captions'][idx,self.num_queries:2*self.num_queries-1]
            t_emb = gtext['class_emb']

            t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
            v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)            

            temperature = self.sem_seg_head.predictor.lang_encoder.logit_scale
            out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)
            
            matched_id = out_prob.max(0)[1]
            mask_pred_results += [pred_gmasks[matched_id,:,:]]

        for i in range(len(mask_pred_results)):
            # upsample masks
            mask_pred_results[i] = F.interpolate(
                mask_pred_results[i][None,],
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bicubic",
                align_corners=False,
                antialias=True
            )[0]

        processed_results = []
        for mask_pred_result, input_per_image, image_size in zip(
            mask_pred_results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results.append({})

            mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                mask_pred_result, image_size, height, width
            )
            processed_results[-1]['grounding_mask'] = mask_pred_result

        return processed_results

    def evaluate_interactive(self, batched_inputs):
        assert len(batched_inputs) == 1, "only support batch size equal to 1"

        # interaction type
        type = batched_inputs[0]['spatial_query']['types'][0]
        assert type in ['point', 'circle', 'scribble', 'polygon', 'box'], "only support point, circle, scribble, polygon, box"

        # getting image 
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, 32)

        # LBK SAM propagation - (1)
        image_embeddings, hier_embeddings_dict = self.backbone.forward_image_embedding(images)
        labels = F.interpolate(batched_inputs[0]['spatial_query']['gt_masks'].unsqueeze(0).float(), size=images.shape[2:]).squeeze(0)
        labels = labels == 1 # int2bool
    
        if type=='point':

            # getitng pos masks from dataloader (Point/Circle/Scribble/Polygon/Box)
            pos_masks = batched_inputs[0]['spatial_query']['rand_shape'].squeeze(1).unbind(0)

            # filtering resize, LBK EDIT
            pos_points = [torch.where(x==True) for x in pos_masks]
            pos_points = [(x[0] * images.shape[2]/batched_inputs[0]['image'].shape[1],
                           x[1] * images.shape[3]/batched_inputs[0]['image'].shape[2]) for x in pos_points]
            pos_points = [(x[0].int(), x[1].int()) for x in pos_points]
            resized_pos_masks = torch.zeros(len(pos_points), images.shape[2], images.shape[3]).cuda()
            for i in range(len(pos_points)): resized_pos_masks[i][pos_points[i]] = 1 
            pos_masks = resized_pos_masks.bool().unbind(0)

            all_batch_shape_iou = []
            for i in range(20):

                if i!=0: pos_points = [torch.where(x==True) for x in pos_masks]

                # LBK transformation: pos_masks to pos_points (smart duplication)
                stack_pos_points = [torch.stack((x[1][0], x[0][0])).unsqueeze(0) for x in pos_points if x[0].shape[0]!=0]
                total_pos_points = torch.cat(stack_pos_points, dim=0)

                assert len(total_pos_points) <= self.num_grids_horizon**2, "only support total pos points limited!"
                q = self.num_grids_horizon**2 // len(total_pos_points)
                r = self.num_grids_horizon**2 % len(total_pos_points)
                sam_point_coords = torch.zeros([self.num_grids_horizon**2, 2]).cuda()
                sam_point_labels = torch.ones([self.num_grids_horizon**2]).cuda()
                sam_point_coords[:q*len(total_pos_points)] = total_pos_points.repeat(q, 1)
                sam_point_coords[q*len(total_pos_points):] = total_pos_points[:r]
                sam_input = [{'point_coords': sam_point_coords, 'point_labels': sam_point_labels}]

                # LBK SAM propagation - (2)
                src_output_features, hyper_in_list = self.backbone.decode_from_embedding(image_embeddings, sam_input)
                outputs = self.sem_seg_head(hier_embeddings_dict, src_output_features, hyper_in_list, target_queries=None)

                # upsample masks
                mask_pred_results = F.interpolate(
                    outputs['pred_masks'],
                    size=labels.shape[1:],
                    mode="bicubic",
                    align_corners=False,
                    antialias=True
                ).squeeze(0)[:self.num_queries-1] > 0

                # hugarian matching
                idx_list = []
                for label in labels: idx_list.append((label == mask_pred_results).sum(dim=(1,2)).argmax())
                pred_masks = mask_pred_results[idx_list, ...]

                # computting ious 
                ious = get_iou(labels, pred_masks)
                all_batch_shape_iou += [ious]
                
                # pos_masks update
                pos_masks = self.prepare_next_spaital_mask(labels, pred_masks, pos_masks)

            all_batch_shape_iou = torch.stack(all_batch_shape_iou)
            processed_results = [{"mask_iou": all_batch_shape_iou[:,i]} for i in range(len(all_batch_shape_iou[0]))]
            return processed_results

        elif type=='box':
            box_points = batched_inputs[0]['spatial_query']['box_points']
            sam_boxes = torch.tensor([list(map(int, box_point)) for box_point in box_points]).cuda()
            

            assert len(sam_boxes) <= 100, "only support total boxes <= 100"
            q = 16**2 // len(sam_boxes)
            r = 16**2 % len(sam_boxes)
            sam_box_coords = torch.zeros([100, 4]).cuda()
            sam_box_coords[:q*len(sam_boxes)] = sam_boxes.repeat(q, 1)
            sam_box_coords[q*len(sam_boxes):] = sam_boxes[:r]
            sam_input = [{'boxes': sam_box_coords}]

            # LBK SAM propagation - (2)
            src_output_features, hyper_in_features = self.backbone.decode_from_embedding(image_embeddings, sam_input)
            outputs = self.sem_seg_head(hier_embeddings_dict, src_output_features, hyper_in_features, target_queries=None)

            # upsample masks
            mask_pred_results = F.interpolate(
                outputs['pred_masks'],
                size=labels.shape[1:],
                mode="bicubic",
                align_corners=False,
                antialias=True
            ).squeeze(0)[:self.num_queries-1] > 0

            # hugarian matching
            idx_list = []
            for label in labels: idx_list.append((label == mask_pred_results).sum(dim=(1,2)).argmax())
            pred_masks = mask_pred_results[idx_list, ...]

            # computting ious 
            ious = get_iou(labels, pred_masks)
            all_batch_shape_iou = torch.stack([ious])
            processed_results = [{"mask_iou": all_batch_shape_iou[:,i]} for i in range(len(all_batch_shape_iou[0]))]
            return processed_results


        elif (type=='circle') or (type=='scribble') or (type=='polygon'):
            all_batch_shape_iou = []

            # LBK transformation: pos_masks to pos_points (smart duplication)
            mask2point = lambda x: torch.stack([torch.tensor([b, a]) for a, b in zip(*x)])[4,:].unsqueeze(0)
            pos_points = [mask2point(torch.where(x==True)) for x in pos_masks]
            total_pos_points = torch.cat(pos_points, dim=0)
            assert len(total_pos_points) <= 100, "only support total pos points <= 100"
            q = 16**2 // len(total_pos_points)
            r = 16**2 % len(total_pos_points)
            sam_point_coords = torch.zeros([100, 2]).cuda()
            sam_point_labels = torch.ones([100]).cuda()
            sam_point_coords[:q*len(total_pos_points)] = total_pos_points.repeat(q, 1)
            sam_point_coords[q*len(total_pos_points):] = total_pos_points[:r]
            sam_input = [{'point_coords': sam_point_coords, 'point_labels': sam_point_labels}]

            # LBK SAM propagation - (2)
            src_output_features, hyper_in_features = self.backbone.decode_from_embedding(image_embeddings, sam_input)
            outputs = self.sem_seg_head(hier_embeddings_dict, src_output_features, hyper_in_features, hyper_in_list, target_queries=None)

            # upsample masks
            mask_pred_results = F.interpolate(
                outputs['pred_masks'],
                size=labels.shape[1:],
                mode="bicubic",
                align_corners=False,
                antialias=True
            ).squeeze(0)[:self.num_queries-1] > 0

            # hugarian matching
            idx_list = []
            for label in labels: idx_list.append((label == mask_pred_results).sum(dim=(1,2)).argmax())
            pred_masks = mask_pred_results[idx_list, ...]

            # computting ious 
            ious = get_iou(labels, pred_masks)
            all_batch_shape_iou += [ious]
            
            # pos_masks update
            pos_masks = self.prepare_next_spaital_mask(labels, pred_masks, pos_masks)

            all_batch_shape_iou = torch.stack(all_batch_shape_iou)
            processed_results = [{"mask_iou": all_batch_shape_iou[:,i]} for i in range(len(all_batch_shape_iou[0]))]
            return processed_results









    def prepare_next_spaital_mask(self, gt_masks, pred_masks, prev_pos_masks):
        
        # dimension unsqueeze
        gt_masks = gt_masks.unsqueeze(0)
        pred_masks = pred_masks.unsqueeze(0)
        pos_masks = torch.cat([i.unsqueeze(0) for i in prev_pos_masks], dim=0)

        # fn: False Negative, gt:1, pred:0 and Filtering
        fn = gt_masks & ~pred_masks
        for id, fnfn in enumerate(fn[0]):
            if (fnfn==1).sum()==0:
                fn[0, id] = gt_masks[0, id]

        # compute iou between gt and pred
        iou = (gt_masks & pred_masks).sum(list(range(2,len(fn.shape)))) / ((gt_masks | pred_masks).sum(dim=list(range(2,len(fn.shape)))) + 1e-8)

        # conv implementation
        bs, ns, h, w = fn.shape
        mask_dt = (distance_transform((~F.pad(fn, pad=(1, 1, 1, 1), mode='constant', value=0)).float())[:,:,1:-1,1:-1]).reshape(bs*ns,-1)
        # idx_criterion = mask_dt.topk(k=10, dim=-1)[1] # (best)
        idx_criterion = torch.cat([(mask_dt[i] > 0).nonzero()[torch.randint(0, len((mask_dt[i] > 0).nonzero()), (1,))][0] for i in range(len(mask_dt))]).cpu() # (best random)
        max_xy_idx = torch.stack([torch.arange(bs*ns), idx_criterion]).tolist()
        next_mask = torch.zeros(gt_masks.shape).cuda().bool()
        next_mask = next_mask.view(bs*ns,-1)
        next_mask[max_xy_idx] = True
        next_mask = next_mask.reshape((bs*ns,1,h,w)).float()
        next_mask = F.conv2d(next_mask, torch.ones((1, 1, 3, 3)).cuda(), padding=1).reshape(bs,ns,h,w) > 0        

        # determine whether next mask is zero
        keep = (iou < 0.925)
        next_masks = next_mask & keep.view(bs,ns,1,1)

        # merged masks
        merged_masks = torch.cat([pos_masks, next_masks.squeeze(0)], dim=0).unbind(0)

        # visualization  
        # gt = gt_masks.squeeze(0).permute(1,2,0).sum(dim=2, keepdim=True).cpu().numpy()
        # fnn = fn.squeeze(0).permute(1,2,0).sum(dim=2, keepdim=True).cpu().numpy()
        # pred_mask = pred_masks.squeeze(0).permute(1,2,0).sum(dim=2, keepdim=True).cpu().numpy()
        # pos_mask = pos_masks.permute(1,2,0).sum(dim=2, keepdim=True).cpu().numpy()
        # new_mask = next_mask.squeeze(0).permute(1,2,0).sum(dim=2, keepdim=True).cpu().numpy()
        # merged_mask = torch.cat([i.unsqueeze(0) for i in merged_masks], dim=0).permute(1,2,0).sum(dim=2, keepdim=True).cpu().numpy()

        return merged_masks


    def prepare_vlp_targets(self, batched_inputs):
        input_ids = []
        attention_mask = []
        for cnt, x in enumerate(batched_inputs):
            captions = x['captions']
            randid = random.randint(0, len(captions)-1)
            input_ids += x['tokens']['input_ids'][randid:randid+1]
            attention_mask += x['tokens']['attention_mask'][randid:randid+1]

        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        tokens = {"input_ids": input_ids, "attention_mask": attention_mask}
        lang_results = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(tokens, token=True)

        target_vlp = []
        for cnt, x in enumerate(batched_inputs):
            target_dict = {}
            target_dict["caption_tokens"] = lang_results['token_emb'][cnt:cnt+1]
            target_dict["caption_proj"] = lang_results['class_emb'][cnt:cnt+1]
            target_dict["caption_tokenids"] = lang_results['tokens']['input_ids'][cnt:cnt+1]
            target_dict["caption_mask"] = lang_results['tokens']['attention_mask'][cnt:cnt+1]            
            target_vlp.append(target_dict)
        return target_vlp

    # LLM 
    def prepare_llm_targets(self, batched_inputs):
        input_ids = []
        attention_mask = []
        labels = []
        for x in batched_inputs:
            captions = x['captions']
            randid = random.randint(0, len(captions)-1)
            input_ids += x['tokens']['input_ids'][randid:randid+1]
            attention_mask += x['tokens']['attention_mask'][randid:randid+1]
            labels += x['tokens']['labels']        
        return self.data_collator({"input_ids": input_ids, "attention_mask": attention_mask, 'labels': labels})

    
    def prepare_targets(self, batched_inputs, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for idx, batch_per_image in enumerate(batched_inputs):
            targets_per_image = batch_per_image["instances"].to(self.device)

            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks

            gt_boxes = targets_per_image.gt_boxes.tensor
            ratio = torch.tensor([w_pad,h_pad,w_pad,h_pad]).to(gt_boxes.device)[None,:]
            gt_boxes = gt_boxes / ratio
            xc,yc,w,h = (gt_boxes[:,0] + gt_boxes[:,2])/2, (gt_boxes[:,1] + gt_boxes[:,3])/2, gt_boxes[:,2] - gt_boxes[:,0], gt_boxes[:,3] - gt_boxes[:,1]
            gt_boxes = torch.stack([xc,yc,w,h]).permute(1,0)

            target_dict = {
                    "labels": targets_per_image.gt_classes,
                    "is_things": targets_per_image.is_things,
                    "masks": padded_masks,
                    "boxes": gt_boxes
                    }

            if self.task_switch['caption']:
                caption = batch_per_image["captions"]
                caption_noun = batch_per_image["captions_noun"]
                rand_index = random.randint(0, len(caption)-1)

                text = caption[rand_index]
                nouns = caption_noun[rand_index]
                noun_captions = [prompt_engineering(noun, topk=10000, suffix='.') for noun in nouns] + [text]
                
                self.sem_seg_head.predictor.lang_encoder.get_text_embeddings(noun_captions, is_eval=False, name='caption_noun', prompt=False)
                ctext = getattr(self.sem_seg_head.predictor.lang_encoder, '{}_text_embeddings'.format('caption_noun'))
                target_dict["captions"] = ctext
                
                target_dict["captions_hash"] = [(hash(st.stem(txt)) % 10**16) for txt in (nouns + [text])]
                target_dict["labels_hash"] = [(hash(st.stem(COCO_PANOPTIC_CLASSES[label_id].replace('-other','').replace('-merged','').replace('-stuff',''))) % 10**16) for label_id in target_dict['labels']]
                
            if self.task_switch['grounding']:
                grd_masks = batch_per_image['groundings']['masks']
                grd_texts = batch_per_image['groundings']['texts']
                grd_hash = batch_per_image['groundings']['hash']
                grd_task = batch_per_image['groundings']['mode']
                
                if len(grd_masks) == 0:
                    padded_masks = None
                else:
                    padded_masks = torch.zeros((grd_masks.shape[0], h_pad, w_pad), dtype=grd_masks.dtype, device=grd_masks.device)
                    padded_masks[:, : grd_masks.shape[1], : grd_masks.shape[2]] = grd_masks

                gtext = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(grd_texts, name='grounding', token=False, norm=False)
                token_emb = gtext['token_emb']
                tokens = gtext['tokens']
                
                unique_hash_id = np.unique(grd_hash, return_index=True)[1]
                selected_mask = np.zeros(len(grd_hash)).astype(np.bool)
                selected_mask[unique_hash_id] = True

                selected_token_emb = token_emb[selected_mask]
                selected_attn_mask = tokens['attention_mask'][selected_mask]
                query_emb = selected_token_emb[selected_attn_mask.bool()]
                
                class_idx = tokens['attention_mask'].sum(dim=-1) - 1
                class_idx = torch.stack((torch.arange(len(class_idx), device=class_idx.device), class_idx)).tolist()
                class_emb = token_emb[class_idx]
                
                target_dict['grounding_masks'] = padded_masks
                target_dict['grounding_query_embs'] = query_emb
                target_dict['grounding_class_embs'] = class_emb
                target_dict['grounding_hash'] = grd_hash
                target_dict['grounding_task'] = grd_task

            new_targets.append(target_dict)
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred, keep_sem_bgd=False):
        if keep_sem_bgd:
            mask_cls = F.softmax(mask_cls, dim=-1)
        else:
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            thing_dataset_id_to_contiguous_id = self.metadata.thing_dataset_id_to_contiguous_id if hasattr(self.metadata, 'thing_dataset_id_to_contiguous_id') else {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )
            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred, box_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)

        labels_per_image = labels[topk_indices]
        topk_indices = (topk_indices // self.sem_seg_head.num_classes)
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]
        if box_pred is not None:
            box_pred = box_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            thing_dataset_id_to_contiguous_id = self.metadata.thing_dataset_id_to_contiguous_id if hasattr(self.metadata, 'thing_dataset_id_to_contiguous_id') else {}
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in thing_dataset_id_to_contiguous_id.values()
            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

            if box_pred is not None:
                box_pred = box_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        # result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)

        if box_pred is not None:
            result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()
        else:
            result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image

        return result



@register_model
def get_xdecoder_model(cfg, **kwargs):
    return GeneralizedXdecoder(cfg)