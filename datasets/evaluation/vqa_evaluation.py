# Copyright (c) Facebook, Inc. and its affiliates.
# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import os
import sys
import json
import copy
import logging
import itertools

import detectron2.utils.comm as comm
from detectron2.evaluation.evaluator import DatasetEvaluator
from collections import OrderedDict

from llm.eval.vqa import VQA
from llm.eval.vqaEval import VQAEval


class VQAEvaluator(DatasetEvaluator):
    """
    Evaluate VQA Accuracy
    """

    def __init__(
        self,
        dataset_name=None,
        distributed=True,
        output_dir=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:
                    "json_file": the path to the COCO format annotation
                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:
                1. "instances_predictions.pth" a file that can be loaded with `torch.load` and
                   contains all the results in the format they are produced by the model.
                2. "coco_instances_results.json" a json file in COCO's result format.
        """
        self._logger = logging.getLogger(__name__)
        self._distributed = distributed
        self._output_dir = output_dir
        self._dataset_name = dataset_name
        self._output_dir = output_dir

    def reset(self):
        self._gen_answers = []
        self._question_ids = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the batch inputs to SysLearner model.
            outputs: the outputs of a SysLearner model. It is a list of dicts with key
                "text" that contains generated answers and "question_id" that contains question ids.
        """
        for output in outputs:
            self._gen_answers.append(output['text'])
            self._question_ids.append(output['question_id'])

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            def gather(x, move=False):
                x = comm.gather(x)
                x = list(itertools.chain(*x))
                if move:
                    x = [xx.to(self._gen_answers[0].device) for xx in x]
                return x
            gen_answers = gather(self._gen_answers)
            question_ids = gather(self._question_ids)
            if not comm.is_main_process():
                return {}
        else:
            gen_answers = self._gen_answers
            question_ids = self._question_ids

        pred_answers = [{"question_id": question_id, "answer": answer} for question_id, answer in zip(question_ids, gen_answers)]
        pred_pth = os.path.join(self._output_dir, '{}_results.json'.format(self._dataset_name))
        json.dump(pred_answers, open(pred_pth, "w"))

        # Evaluate on EvalAI server
        if 'test' in self._dataset_name:
            return

        annFile = '/mnt/ssd/lbk-cvpr/dataset/VQAv2/v2_mscoco_val2014_annotations.json'
        quesFile = '/mnt/ssd/lbk-cvpr/dataset/VQAv2/v2_OpenEnded_mscoco_val2014_questions.json'

        # create vqa object and vqaRes object
        vqa = VQA(annFile, quesFile)
        vqaRes = vqa.loadRes(pred_pth, quesFile)

        # create vqaEval object by taking vqa and vqaRes
        vqaEval = VQAEval(vqa, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2

        # evaluate results
        """
        If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
        By default it uses all the question ids in annotation file
        """
        vqaEval.evaluate() 

        self._results = OrderedDict()
        # Copy so the caller can do whatever with results
        self._results['accuracy'] = vqaEval.accuracy['overall']
        self._results['perQuestionType'] = {}
        for quesType in vqaEval.accuracy['perQuestionType']:
            self._results['perQuestionType'][quesType] = vqaEval.accuracy['perQuestionType'][quesType]

        self._results['perAnswerType'] = {}
        for ansType in vqaEval.accuracy['perAnswerType']:
            self._results['perAnswerType'][ansType] = vqaEval.accuracy['perAnswerType'][ansType]
        
        return copy.deepcopy(self._results)