import argparse
import glob
import json
import logging
import math
import os
import random
import re
import sys

import jieba
import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_restful import Api, Resource, reqparse
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from creeper import creeper_v1, creeper_v2
from creeper.spider import crawl
from mrc import mrc_MODEL_CLASSES, mrc_predict, set_seed, to_list
from rerank import rerank_MODEL_CLASSES, rerank_predict
from models import BertForBaiduQA_Answer_Selection

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Args(object):
    def __init__(self, config: dict):
        for key, value in config.items():
            self.__dict__[key] = value


class Mrc(object):
    """
    ADD KEYS:
    answer: string
    mrc_logits: float 
    """

    def __init__(self, config: dict):
        """
        Loading args, model, tokenizer
        """
        logger.info("***** Mrc model initing *****")
        args = Args(config['mrc'])

        # Setup CUDA, GPU & distributed training
        if args.local_rank == -1 or args.no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
            args.n_gpu = torch.cuda.device_count()
        else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            torch.distributed.init_process_group(backend='nccl')
            args.n_gpu = 1
        args.device = device

        # Set seed
        set_seed(args)

        # Load pretrained model and tokenizer
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        # model_name_or_path: the path of pre-trained_model or checkpoint
        args.model_type = args.model_type.lower()
        config_class, model_class, tokenizer_class = mrc_MODEL_CLASSES[args.model_type]
        self.config = config_class.from_pretrained(args.model_name_or_path)
        self.tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
        # self.model = model_class.from_pretrained(args.model_name_or_path,
        #                                          from_tf=bool('.ckpt' in args.model_name_or_path), config=self.config)
        state_dict  = torch.load(args.model_name_or_path+'/pytorch_model.bin',map_location='cpu')
        self.model = BertForBaiduQA_Answer_Selection(config= self.config)
        self.model.load_state_dict(state_dict= state_dict)
        
        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        self.model.to(args.device)
        self.args = args

        logger.info("Training/evaluation parameters %s", args)

    def predict(self, examples):
        """
        all_predictions:
        {
            'question_id': answer string,
            'question_id': answer string,
            ...
            ...
        }
        all_nbest_json:
        {
            'question_id':[
                {
                    "text": string,
                    "probability": float,
                    "start_logit": float,
                    "end_logit": float
                },
                {
                    "text": string,
                    "probability": float,
                    "start_logit": float,
                    "end_logit": float
                },
                ...(top 20)
            ]
        }

        examples:
        [
            {
                'question_id': int,
                'question': string,
                'title': string,
                'abstract': string,
                'source_link': url,
                'content': string,
                'doc_tokens': string,
            },f
            {
            ...
            }
        ]       
        """
        answers = mrc_predict(self.args, self.model, self.tokenizer, examples)

        # all_predictions, all_nbest_json = mrc_predict(self.args, self.model, self.tokenizer, examples)
        # assert len(all_predictions) == len(examples)
        # assert len(all_nbest_json) == len(examples)
        # for example in examples:
        #     qid = example['question_id']
        #     logitslist = [var['start_logit'] + var['end_logit'] for var in all_nbest_json[qid]]
        #     problist = [var['start_prob'] * var['end_prob'] for var in all_nbest_json[qid]]
        #     problist_v1 = [var['start_prob_v1'] * var['end_prob_v1'] for var in all_nbest_json[qid]]
        #     example['answer'] = all_predictions[qid].replace('\n', '').replace(' ', '').strip()
        #     example['answer_start_index'] = all_nbest_json[qid][0]['start_index']
        #     example['answer_end_index'] = all_nbest_json[qid][0]['end_index']
        #     example['raw_text']=all_nbest_json[qid][0]['raw_text']
        #     example['mrc_logits'] = sum(logitslist) / len(logitslist)
        #     example['mrc_prob'] = sum(problist) / len(problist)
        #     example['mrc_prob_v1'] = sum(problist_v1) / len(problist_v1)
        return answers

class Demo(object):
    def __init__(self, config_path):
        self.server_config = json.loads(open(config_path).read())
        self.mrc_processor = Mrc(self.server_config)
        # if self.server_config["creeper"]["creeper_type"] == 'v1':
        #     self.creeper = creeper_v1
        # else:
        #     self.creeper = creeper_v2
        # self.creeper = crawl
        self.keys = [
            "answer"
        ]
        self.indexs = [
            "answer",
            "answer_start_index",
            "answer_end_index",
            "raw_text",
            "question_id"

        ]
        # self.keys = [
        #     "question_id",
        #     "question",
        #     "title",
        #     "abstract",
        #     "source_link",
        #     "content",
        #     "answer",
        #     "final_prob",
        #     "final_prob_v1"
        # ]

    def filter(self, examples, keys):
        new_examples = []
        for example in examples:
            new_example = {}
            for key in example:
                if key in keys:
                    new_example[key] = example[key]
            new_examples.append(new_example)
        return new_examples

    def predict(self, query):
        # examples = self.creeper(query)
        #print("part1 Finish")
        examples = []
        examples = self.mrc_processor.predict(examples)
        #print("part2 Finish")
        # examples = self.rerank_processor.predict(examples)
        #print("part3 Finish")
        # examples = self.choose_processor.process(examples)
        # if( examples[0]['doc_tokens']== examples[0]['temp_tokens']):
        #     examples = self.filter(examples, self.indexs)
        #     return examples[0]
        # examples[0]['doc_tokens']= examples[0]['temp_tokens']
        # examples.append(self.mrc_processor.predict([examples[0]])[0])
        # examples = self.rerank_processor.predict(examples)
        # examples = self.choose_processor.process(examples)
        # examples = self.filter(examples, self.indexs)
        #print("part5 Finish")
        return examples


if __name__ == "__main__":
    # app = Flask(__name__)

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config_path", default=None, type=str, required=True,
    #                     help="config json")
    # parser.add_argument("--port", default=None, type=int, required=True,
    #                     help="config json")
    # args = parser.parse_args()
    # print("____________________________________"+ args.config_path)
    D = Demo("my_config.json")
    answers = D.predict("西红柿做法")
    with open('data/preprocessed/my_dev/dev_pre.json','w',encoding='utf-8',newline='\n') as w:
        for ans in answers:
            w.write(json.dumps(ans,ensure_ascii=False)+'\n')


    # @app.route('/api/func1', methods=['POST', 'GET'])
    # def func1():
    #     try:
    #         if request.method == 'POST':
    #             inputs = request.get_json(force=True)
    #             query = inputs['query']
    #         else:
    #             query = request.args.get('query')
    #        # return json.dumps({'code': 0, 'results': D.predict(query)}, ensure_ascii=False)
    #         return json.dumps(D.predict(query), ensure_ascii=False)
    #     except Exception as e:
    #             return json.dumps({'code': 1, 'messge': str(e)})


    # app.run(host="0.0.0.0", port=args.port, threaded=True)
