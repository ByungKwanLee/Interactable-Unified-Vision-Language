# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------
from tqdm import tqdm
from datetime import datetime
import time
import os
import sys
import importlib
import json
import random
import wandb
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
from .utils_trainer import UtilsTrainer
from .utils.misc import *
from .utils.serialization import JSONEncoder, filter_jsonable

logger = logging.getLogger(__name__)


class DefaultTrainer(UtilsTrainer, DistributedTrainer):

    def __init__(self, opt):
        """
        Set up the task the model is being trained for.
        """
        super().__init__(opt)
        base_name = 'base_dir'
        base_path =  os.path.join(self.opt['base_path'], '__init__.py')
        spec = importlib.util.spec_from_file_location(base_name, base_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[base_name] = module
        spec.loader.exec_module(module)
        logger.info(f"Imported {base_name} at base_path {self.opt['base_path']}")

        pipeline_module = importlib.import_module(f"base_dir.pipeline.{self.opt['PIPELINE']}")
        pipeline_class = getattr(pipeline_module, self.opt['PIPELINE'])
        logger.info(f"Pipeline for training: {self.opt['PIPELINE']}")
        self.pipeline = pipeline_class(self.opt)

    def eval(self, ):
        logger.info('-----------------------------------------------')
        logger.info("Evaluating model ... ")
        self.mode = "eval"

        # self.model_names, self.raw_models, self.criteria = self.pipeline.set_up_model()
        self.raw_models = self.pipeline.initialize_model()
        self.model_names = self.raw_models.keys()

        # move models to the device
        for module_name in self.model_names:
            self.raw_models[module_name].to(self.opt['device'])

        # load model during evaluation
        if self.opt['WEIGHT'] and os.path.isfile(self.opt['RESUME_FROM']):
            model_path = self.opt['RESUME_FROM'] 
            self.load_model(model_path)
        else:
            raise ValueError(f"Model not found: {model_path}")

        results = self._eval_on_set()
        if self.opt['rank'] == 0: self.dictionary_display(results)

    def _eval_on_set(self):
        # logger.info(f"Evaluation start ...")
        if self.opt['FP16']:
            from torch.cuda.amp import autocast
            with autocast():
                results = self.pipeline.evaluate_model(self)
        else:        
            results = self.pipeline.evaluate_model(self)
        # if self.opt['rank'] == 0:
        #     logger.info(results)
        return results

    def compute_loss(self, forward_func, batch):
        if self.opt['FP16']:
            from torch.cuda.amp import autocast
            with autocast():
                loss = forward_func(self, batch)
        else:
            loss = forward_func(self, batch)
        return loss
    
    def backward_loss(self, loss):  # noqa: E252

        if self.grad_acc_steps > 1:
            loss = loss / self.grad_acc_steps

        if self.opt['FP16']:
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()
        return loss

    def update_model(self, model_name='default'):
        if self.opt['FP16']:
            self.grad_scaler.unscale_(self.optimizers[model_name])
            self.grad_scaler.step(self.optimizers[model_name])
        else:
            self.optimizers[model_name].step()

        self.optimizers[model_name].zero_grad()
        self.train_params['optim_steps'][model_name] += 1
        self.lr_schedulers[model_name].step()

    def train_step(self, batch):
        self.grad_acc_batches.append(batch) # support batch accumulation

        if self.is_gradient_accumulation_boundary():
            # set all modules and criteria into training mode
            for model_name in self.model_names:
                self.models[model_name].train()

            assert len(self.grad_acc_batches) == self.grad_acc_steps

            total_batch_sample = 0
            for batch in self.grad_acc_batches:

                # pipeline/XDecoderPipeline.py
                loss_info, sample_size_info = self.pipeline.forward_step(self, batch)

                self.train_loss.update_iter(loss_info)
                total_batch_sample += sample_size_info['num_samples']

            if self.opt['FP16']:
                # Update GradScaler after an effective batch
                self.grad_scaler.update()

            # update losses and item counts of an effective batch to the AverageMeters
            if self.opt['world_size'] > 1:
                total_batch_sample = torch.tensor(total_batch_sample).to(self.opt['device'])
                torch.distributed.all_reduce(total_batch_sample, torch.distributed.ReduceOp.SUM)
                total_batch_sample = total_batch_sample.item()

            self.train_params['total_batch_size'] += total_batch_sample
            self.grad_acc_batches = []

        self.train_params['num_updates'] += 1
        
    def init_train(self):
        self.mode = "train"
        logger.info('-------------------------------------------------------')
        logger.info("Training on rank: {}".format(self.opt['rank']))

        self.raw_models = self.pipeline.initialize_model()
        self.model_names = list(self.raw_models.keys())

        # move models to the device
        for module_name in self.model_names:
            self.raw_models[module_name].to(self.opt['device'])

        self.train_dataloaders = self.pipeline.get_dataloaders('train', is_evaluation=False)
        self.train_params = {
                             "updates_per_epoch": len(self.train_dataloaders),
                             "total_batch_size": 0,
                             "num_updates": 0,
                             "optim_steps": {module_name: 0 for module_name in self.model_names},
                             "start_epoch_idx": 0,
                             "start_batch_idx": 0,
                             "current_epoch_idx": 0,
                             "current_batch_idx": 0,
                             "resume_epoch_idx": 0, 
                             }

        self.train_loss = LossMeter()
        self.grad_acc_batches = []

        if self.opt['CUDA']:
            torch.cuda.empty_cache()

        self.create_optimizer_and_scheduler()
        self.models = {model_name: self.raw_models[model_name] for model_name in self.model_names}
        self._initialize_ddp()

        if self.opt.get('WEIGHT', False):
            self.load_weight(self.opt['RESUME_FROM'], must_exist=True)
        if self.opt.get('RESUME', False):
            self.load_checkpoint(self.opt['RESUME_FROM'], must_exist=True)

        ######################
        # Start the main loop
        ######################
        if self.opt['rank'] == 0:
            # Train!
            logger.info("***** Running training *****")
            logger.info(f"  Num of GPUs = {self.opt['world_size']}")
            logger.info(f"  Num Epochs = {self.opt['SOLVER']['MAX_NUM_EPOCHS']}")
            logger.info(f"  Num of Mini Batches per Epoch = {self.train_params['updates_per_epoch']}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.opt['SOLVER']['MAX_NUM_EPOCHS'] * self.train_params['updates_per_epoch']}")
            logger.info(f"  Gradient Accumulation steps = {self.grad_acc_steps}")
            logger.info(f"  Total optimization steps = {self.opt['SOLVER']['MAX_NUM_EPOCHS'] * self.train_params['updates_per_epoch'] // self.grad_acc_steps}")

    @staticmethod
    def dictionary_display(results):
        print('\n-------------------')
        for key, value in results.items():
            print(f'DATASET/Task: [{key}]\n')
            for _key, _value in value.items():
                print(f'{_key}:')
                for __key, __value in _value.items():
                    print(f'    {__key}: {__value}')
            print('-------------------')
        print('\n')

    def train(self):
        """
        Training
        """
        self.init_train()
        current_optim_steps = self._get_and_validate_current_optim_steps()
        num_epochs = self.opt['SOLVER']['MAX_NUM_EPOCHS']

        if self.opt.get('EVAL_AT_START', False):
            results = self._eval_on_set()
            if self.opt['rank'] == 0 and self.opt['WANDB']:
                wandb.log(results)

        max_length_dataset = 0
        for dataset_name in self.train_dataloaders.dataset_names:
            if max_length_dataset < len(getattr(self.train_dataloaders, dataset_name)):
                max_length_dataset = len(getattr(self.train_dataloaders, dataset_name))
        
        for epoch in range(self.train_params['start_epoch_idx'], num_epochs):
            self.train_params['current_epoch_idx'] = epoch
            if self.opt['rank'] == 0: print(f"Start epoch: {epoch} training.")
            
            prog_bar = tqdm(enumerate(self.train_dataloaders), total=max_length_dataset, leave=True)
            for batch_idx, batch in prog_bar:
                self.train_params['current_batch_idx'] = batch_idx
                prev_optim_steps = current_optim_steps

                # update
                self.prev_optim_steps = prev_optim_steps
                self.train_step(batch)

                current_optim_steps = self._get_and_validate_current_optim_steps()
                
                last_lr = {}
                for module_name in self.model_names:
                    last_lr[module_name] = self.lr_schedulers[module_name].get_last_lr()[0]
                
                if self.opt['rank'] == 0:
                    if self.opt['WANDB']:
                        # log for wandb
                        wb_loss_info = {key: obj.val for key, obj in self.train_loss.losses.items()}
                        wandb.log(wb_loss_info, step=self.prev_optim_steps)

                loss_list = [obj.val for _, obj in self.train_loss.losses.items()]
                total_loss = sum(loss_list) / len(loss_list)
                desc = f"|Epochs[{epoch}]|[{batch_idx+1}/{max_length_dataset}]|"
                desc += f"LR[{', '.join([f'{val:.2e}' for _, val in last_lr.items()])}]|"
                desc += f"Loss[{total_loss:.2f}]|"
                prog_bar.set_description(desc, refresh=True)
                if max_length_dataset == batch_idx + 1: break

            # synchronize
            if torch.cuda.is_available(): torch.cuda.synchronize()

            # evaluate and save ckpt every epoch
            if self.opt['rank'] == 0: print('\n-----------Saving CKPT...-----------\n')
            self.save_checkpoint(self.train_params['num_updates'])
            results = self._eval_on_set()
            if self.opt['rank'] == 0: self.dictionary_display(results)
            if self.opt['rank'] == 0 and self.opt['WANDB']: wandb.log(results)