# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

from datetime import datetime
import time
import os
import sys
import importlib
import json
import random
import logging
import numpy as np
import copy
import contextlib
import shutil
from typing import Any, Callable, Union
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from mpi4py import MPI
from infinibatch import iterators

from .distributed_trainer import DistributedTrainer
from .utils.misc import *
from .utils.serialization import JSONEncoder, filter_jsonable
from utils.distributed import get_world_size

logger = logging.getLogger(__name__)


class UtilsTrainer(DistributedTrainer):

    def __init__(self, opt):
        super().__init__(opt)

    def get_batch_size(self, batch, module_name='default'):
        if hasattr(self.raw_models[module_name], 'get_batch_size'):
            if callable(self.raw_models[module_name].get_batch_size):
                return self.raw_models[module_name].get_batch_size(batch)
        return {}

    def _initialize_ddp(self):
        if self.opt['FP16']:
            from torch.cuda.amp import GradScaler
            self.grad_scaler = GradScaler()
            # logger.warning("PyTorch AMP GradScaler initialized.")

        for module_name in self.model_names:
            if self.opt['world_size'] > 1:
                # ddp: wrap modules for distributed data parallel training
                self.models[module_name] = nn.parallel.DistributedDataParallel(self.models[module_name],
                                                        device_ids=[self.opt['local_rank']],
                                                        output_device=self.opt['local_rank'],
                                                        find_unused_parameters=self.opt.get('FIND_UNUSED_PARAMETERS', True))

    def load_model(self, load_path):
        for module_name in self.model_names:
            self.raw_models[module_name] = self.raw_models[module_name].from_pretrained(load_path)
            self.raw_models[module_name].to(self.opt['device'])

    def save_checkpoint(self):
        
        save_dir = self.save_folder

        if self.opt['world_size'] > 1:
            torch.distributed.barrier()

        if self.opt['rank'] == 0:
            os.makedirs(self.save_folder, exist_ok=True)

        if self.opt['world_size'] > 1:
            torch.distributed.barrier()

        if self.opt['rank'] == 0:
            os.makedirs(save_dir, exist_ok=True)

        if self.opt['rank'] == 0:
            if self.opt['FP16']:
                amp_state = self.grad_scaler.state_dict()
            else:
                amp_state = None
            for module_name in self.model_names:
                module_save_dir = os.path.join(save_dir, module_name)
                os.makedirs(module_save_dir, exist_ok=True)
                save_path = os.path.join(module_save_dir, 'module_training_states.pt')
                state = {'module': self.models[module_name].state_dict(),
                            'optimizer': self.optimizers[module_name].state_dict(),
                            'lr_scheduler': self.lr_schedulers[module_name].state_dict(),
                            'amp_state': amp_state,}
                torch.save(state, save_path)

        if self.opt['rank'] == 0:
            for module_name in self.model_names:
                module_save_dir = os.path.join(save_dir, module_name)
                self.raw_models[module_name].save_pretrained(module_save_dir)

    def load_weight(self, checkpoint_path=None):
        self.load_model(checkpoint_path)

    def load_checkpoint(self, checkpoint_path=None):
        logger.warning(f'Resuming checkpoint from {checkpoint_path}...')

        for model_name in self.model_names:
            model_load_path = os.path.join(checkpoint_path, model_name, 'module_training_states.pt')
            state = torch.load(model_load_path, map_location=self.opt['device'])
            
            logger.warning(f'HACK to strip module from model state dict on single gpu debugging!')
            ckpt = state['module']
            if get_world_size() <= 1:
                ckpt = {key.replace('module.',''):ckpt[key] for key in ckpt.keys()}
                
            self.models[model_name].load_state_dict(ckpt)
            self.optimizers[model_name].load_state_dict(state['optimizer'])
            self.lr_schedulers[model_name].load_state_dict(state['lr_scheduler'])
            if self.opt['FP16']:
                self.grad_scaler.load_state_dict(state['amp_state'])

        load_path = os.path.join(checkpoint_path, 'trainer_states.pt')
        trainer_state = torch.load(load_path, map_location='cpu')
        self.train_loss = trainer_state['train_loss']

        logger.warning(f'Finished loading checkpoint from {checkpoint_path}.')