# -*- coding: utf-8 -*-
""" Finetuning the library models for question-answering on Duqa (Bert)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import sys

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (WEIGHTS_NAME, AdamW, BertConfig, BertTokenizer,get_linear_schedule_with_warmup)
# from transformers import (WEIGHTS_NAME, AdamW, AlbertConfig, AlbertTokenizer,get_linear_schedule_with_warmup)

from models import BertForBaiduQA_Answer_Selection
from .utils_duqa import (RawResult, convert_examples_to_features, #.utils_duqa
                         convert_output, read_baidu_examples,
                         read_baidu_examples_pred, write_predictions)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForBaiduQA_Answer_Selection, BertTokenizer),
    # 'albert':(AlbertConfig,BertForBaiduQA_Answer_Selection,BertTokenizer)
}
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def main():
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    model = model_class.from_pretrained('checkpoints/mrc_1030')
    tokenizer = tokenizer_class.from_pretrained('checkpoints/mrc_1030', do_lower_case=True)
    model.to('cuda:0')
    examples = read_baidu_examples(input_file='data/preprocessed/my_test/test_v.json', is_training=True)
    features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                doc_stride=args.doc_stride,
                                                max_query_length=args.max_query_length,
                                                is_training=not evaluate)