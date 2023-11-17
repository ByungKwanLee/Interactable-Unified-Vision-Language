import logging
import os
import time
import pickle
import torch
import torch.nn as nn

from utils.distributed import is_main_process

logger = logging.getLogger(__name__)


NORM_MODULES = [
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
    # NaiveSyncBatchNorm inherits from BatchNorm2d
    torch.nn.GroupNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
    torch.nn.LayerNorm,
    torch.nn.LocalResponseNorm,
]

def register_norm_module(cls):
    NORM_MODULES.append(cls)
    return cls

def align_and_update_state_dicts(model_state_dict, ckpt_state_dict):
    model_keys = sorted(model_state_dict.keys())
    ckpt_keys = sorted(ckpt_state_dict.keys())
    result_dicts = {}
    matched_log = []
    unmatched_log = []
    unloaded_log = []

    filtered_model_keys = list(filter(lambda x: not x.startswith('llms.'), model_keys))
    filtered_ckpt_keys = list(filter(lambda x: not x.startswith('backbone.'), ckpt_keys))

    for model_key in filtered_model_keys:
        model_weight = model_state_dict[model_key]
        if model_key in filtered_ckpt_keys:
            ckpt_weight = ckpt_state_dict[model_key]
            if model_weight.shape == ckpt_weight.shape:
                result_dicts[model_key] = ckpt_weight
                filtered_ckpt_keys.pop(filtered_ckpt_keys.index(model_key))
                matched_log.append("Loaded {}".format(model_key))
            else:
                unmatched_log.append("*UNMATCHED* {}".format(model_key))
        else:
            unloaded_log.append("*UNLOADED* {}".format(model_key))
            
    # [print(x) for x in matched_log]
    # [print(x) for x in unmatched_log]
    # [print(x) for x in unloaded_log]
    # exit(0)
    return result_dicts