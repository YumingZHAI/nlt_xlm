# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import argparse
import torch
import numpy as np
import random

from src.utils import bool_flag, initialize_exp
from src.evaluation.phraseTT import phraseTT
from src.model.embedder import SentenceEmbedder

# all GLUE tasks are monolingual English pairs
TASKS = ['nlT']  # non-literal translation

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# parse parameters
parser = argparse.ArgumentParser(description='Train on nlT (literal or non-literal phrase translation)')

# main parameters
parser.add_argument("--exp_name", type=str, default="",
                    help="Experiment name")
parser.add_argument("--dump_path", type=str, default="",
                    help="Experiment dump path")
parser.add_argument("--exp_id", type=str, default="",
                    help="Experiment ID")

# evaluation task / pretrained model
parser.add_argument("--transfer_tasks", type=str, default="",
                    help="Transfer tasks, example: 'MNLI-m,RTE,XNLI' ")
parser.add_argument("--model_path", type=str, default="",
                    help="Model location")

# data
parser.add_argument("--data_path", type=str, default="",
                    help="Data path")
parser.add_argument("--max_vocab", type=int, default=-1,
                    help="Maximum vocabulary size (-1 to disable)")
parser.add_argument("--min_count", type=int, default=0,
                    help="Minimum vocabulary count")
parser.add_argument("--clf_result", type=str, default="",
                    help="File containing the classifier's probability after applying softmax")
parser.add_argument("--trainCorpus", type=str, default="",
                    help="The name of training corpus")
parser.add_argument("--devCorpus", type=str, default="",
                    help="The name of validation corpus")
parser.add_argument("--testCorpus", type=str, default="",
                    help="The name of test corpus")

# batch parameters
parser.add_argument("--max_len", type=int, default=256,
                    help="Maximum length of sentences (after BPE)")
parser.add_argument("--group_by_size", type=bool_flag, default=False,
                    help="Sort sentences by size during the training")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Number of sentences per batch")
parser.add_argument("--max_batch_size", type=int, default=0,
                    help="Maximum number of sentences per batch (used in combination with tokens_per_batch, 0 to disable)")
parser.add_argument("--tokens_per_batch", type=int, default=-1,
                    help="Number of tokens per batch")

# model / optimization
parser.add_argument("--finetune_layers", type=str, default='0:_1',
                    help="Layers to finetune. 0 = embeddings, _1 = last encoder layer")
parser.add_argument("--weighted_training", type=bool_flag, default=False,
                    help="Use a weighted loss during training")
parser.add_argument("--dropout", type=float, default=0,
                    help="Fine-tuning dropout")
parser.add_argument("--optimizer_e", type=str, default="adam,lr=0.0001",
                    help="Embedder (pretrained model) optimizer")
parser.add_argument("--optimizer_p", type=str, default="adam,lr=0.0001",
                    help="Projection (classifier) optimizer")
parser.add_argument("--n_epochs", type=int, default=100,
                    help="Maximum number of epochs")
parser.add_argument("--epoch_size", type=int, default=-1,
                    help="Epoch size (-1 for full pass over the dataset)")
parser.add_argument("--checkpoint_dir", type=str, default="",
                    help="Checkpoint directory path")
parser.add_argument("--inference", type=bool_flag, default=False,
                    help="Use the saved best model for inference")
parser.add_argument("--resume_training", type=bool_flag, default=False,
                    help="Use the saved best model to resume training")
parser.add_argument("--fold", type=int, default=1,
                    help="The index of fold to train on (1-10)")
parser.add_argument("--bestModel", type=str, default="",
                    help="Name for the saved best model (checkpoint)")

# debug
parser.add_argument("--debug_train", type=bool_flag, default=False,
                    help="Use valid sets for train sets (faster loading)")
parser.add_argument("--debug_slurm", type=bool_flag, default=False,
                    help="Debug multi-GPU / multi-node within a SLURM job")

# parse parameters == arguments
# params is a namespace here
params = parser.parse_args()

if params.tokens_per_batch > -1:
    params.group_by_size = True

# check parameters
assert os.path.isdir(params.data_path)
assert os.path.isfile(params.model_path)

# tasks
params.transfer_tasks = params.transfer_tasks.split(',')
assert len(params.transfer_tasks) > 0
assert all([task in TASKS for task in params.transfer_tasks])

# reload a pre-trained model. this is a static method
# /Users/yumingzhai/PycharmProjects/Coling2020/xlm-code/XLM-master/src/model/embedder.py
# load this model: /models/mlm_tlm_xnli15_1024.pth
embedder = SentenceEmbedder.reload(params.model_path, params)

# reload langs from pretrained model (15 here, including EN, FR, ZH)
params.n_langs = embedder.pretrain_params['n_langs']
params.id2lang = embedder.pretrain_params['id2lang']
params.lang2id = embedder.pretrain_params['lang2id']

# initialize the experiment: dump parameters, create a logger
logger = initialize_exp(params)
scores = {}

# prepare trainers / evaluators
phrasett = phraseTT(embedder, scores, params)

# run
phrasett.run()
