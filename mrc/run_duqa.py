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
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,3'
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForBaiduQA_Answer_Selection, BertTokenizer),
    # 'albert':(AlbertConfig,BertForBaiduQA_Answer_Selection,BertTokenizer)
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(args, train_dataset, model, tokenizer):
    """ Train the model """

    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=16,pin_memory=True)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    # 感觉这里是在把预训练模型的权重读入进来
    # for name, param in model.named_parameters():
	#     print(name,param.requires_grad)
	#     if('bert2.bert' in name):
    #         param.requires_grad = False
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'bert2.bert'not in n], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)and 'bert2.bert'not in n], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        model.to(args.device)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0            #——————————————————————from here different
    tr_loss, logging_loss = 0.0, 0.0  # tr loss 是啥不知道，可能是交叉熵
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])   #local rank 也不是知道是啥
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    # f = open('o6.txt','a+',encoding='utf-8')
    # ff = open('each_loss2.txt','a+',encoding='utf-8')
    for epoch_idx, epoch in tqdm(enumerate(train_iterator), desc='training epoches'):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        # if epoch_idx < 3:
        #         continue  # temp
        for step, batch in tqdm(enumerate(epoch_iterator), desc='training batches'):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'q_input_ids':       batch[0],
                      'q_attention_mask':  batch[1], 
                      'q_token_type_ids':  batch[2],  
                      'p_input_ids':       batch[3],
                      'p_attention_mask':  batch[4], 
                      'p_token_type_ids':  batch[5],  
                      'start_positions':   batch[6], 
                      'end_positions':     batch[7],
                      'right_num':         batch[8]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)  
            with open('final2.txt','a+',encoding='utf-8') as f:
                f.write(str(loss.mean())+'------'+str(epoch_idx)+'\n') 
            # ff.write(str(outputs[1].mean())+'---1---'+str(epoch_idx)+'\n') 
            # ff.write(str(outputs[2].mean())+'---2---'+str(epoch_idx)+'\n') 
            with open('final_eachloss3.txt','a+',encoding='utf-8') as ff:
                ff.write(str(outputs[3].mean())+'---3---'+str(epoch_idx)+'\n') 
            # with open('true_train_op_detail.txt','a+',encoding='utf-8') as f:       
            #     f.write(str(float(outputs[1]))+str(float(outputs[2]))+str(float(outputs[3]))+'------'+str(epoch_idx)+'\n') 
            # 这个时候output出来的 是loss-> Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()  # 更新模型
                scheduler.step()  # 更新学习率
                model.zero_grad() # 模型的梯度置0
                global_step += 1  # 加一步

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

        if args.local_rank in [-1, 0]:
            # Save model checkpoint by epoch
            output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(epoch_idx))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logger.info("Saving model checkpoint to %s", output_dir)

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step



def evaluate(args, model, tokenizer, prefix=""):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_results = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2]
                      }
            example_indices = batch[3]
            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            result = RawResult(unique_id    = unique_id,
                                start_logits = to_list(outputs[0][i]),
                                end_logits   = to_list(outputs[1][i]))     
            all_results.append(result)

    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))
    write_predictions(examples, features, all_results, args.n_best_size,
                    args.max_answer_length, args.do_lower_case, output_prediction_file,
                    output_nbest_file, args.verbose_logging)
    return 0

def out_result(output,features,num,ans,predict_examples):
    for i,each_ans in enumerate(output):
        p_example = predict_examples[num+i]
        temp_ans ={}
        feature = features[num+i]
        l = len(feature.token_to_orig_map[each_ans['id']])
        s = min(l,int(each_ans['start'])+1)
        e = min(l,int(each_ans['end'])+2)
        real_start = feature.token_to_orig_map[each_ans['id']][s]
        real_end = feature.token_to_orig_map[each_ans['id']][e]
        temp_ans['question_type'] = p_example.question_type
        temp_ans['question'] = p_example.question_text
        temp_ans['question_id'] = p_example.qas_id
        temp_ans['answers'] = [''.join(p_example.documents[each_ans['id']]['doc_tokens'][real_start:real_end])]
        temp_ans['source'] = 'search'
        temp_ans['score'] =each_ans['score']
        ans.append(temp_ans)

def predict(args, model, tokenizer, raw_data):

    # predict_examples = read_baidu_examples_pred(raw_data, is_training=False)
    predict_examples = read_baidu_examples('data/preprocessed/my_dev/dev.json',is_training=False)
    cached_features_file = "data/preprocessed/my_test/cached_dev"
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        features = convert_examples_to_features(
            examples=predict_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=False
        )
        logger.info("Saving features into cached file %s", "data/preprocessed/my_test/cached_dev")
        torch.save(features, cached_features_file)
    # all_input_ids = torch.tensor([f.input_ids for f in predict_features], dtype=torch.long)
    # all_input_mask = torch.tensor([f.input_mask for f in predict_features], dtype=torch.long)
    # all_segment_ids = torch.tensor([f.segment_ids for f in predict_features], dtype=torch.long)
    
    all_q_input_ids = torch.tensor([f.q_input_ids for f in features], dtype=torch.long)
    all_q_input_mask = torch.tensor([f.q_input_mask for f in features], dtype=torch.long)
    all_q_segment_ids = torch.tensor([f.q_segment_ids for f in features], dtype=torch.long)
    all_p_input_ids = torch.tensor([f.p_input_ids for f in features], dtype=torch.long)
    all_p_input_mask = torch.tensor([f.p_input_mask for f in features], dtype=torch.long)
    all_p_segment_ids = torch.tensor([f.p_segment_ids for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_q_input_ids.size(0), dtype=torch.long)
    dataset = TensorDataset(all_q_input_ids, all_q_input_mask, all_q_segment_ids, 
                            all_p_input_ids, all_p_input_mask, all_p_segment_ids,all_example_index)   

    args.predict_batch_size = args.per_gpu_predict_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    predict_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    predict_dataloader = DataLoader(dataset, sampler=predict_sampler, batch_size=args.predict_batch_size)

    # Predict!
    logger.info("***** Running prediction *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.predict_batch_size)
    # all_results = []
    f_cnt = 0
    ans = []
    for batch in tqdm(predict_dataloader, desc="Predicting"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'q_input_ids':       batch[0],
                      'q_attention_mask':  batch[1], 
                      'q_token_type_ids':  batch[2],  
                      'p_input_ids':       batch[3],
                      'p_attention_mask':  batch[4], 
                      'p_token_type_ids':  batch[5],  
                      }
            # example_indices = batch[6]
            outputs = model(**inputs)
            # if f_cnt>40:
            #     break
        # for i, example_index in enumerate(example_indices):
        #     eval_feature = features[example_index.item()]
        #     unique_id = int(eval_feature.unique_id)
        #     result = RawResult(unique_id    = unique_id,
        #                         start_logits = to_list(outputs[0][i]),
        #                         end_logits   = to_list(outputs[1][i]))     
        #     all_results.append(result)
            # num = f_cnt* args.predict_batch_size
            # for output in outputs:
            #     out_result(output,features,num,ans,predict_examples)
            for i,each_ans in enumerate(outputs):
                # if f_cnt == 114:
                #     f_cnt = 114
                p_example = predict_examples[f_cnt* args.predict_batch_size+i]
                temp_ans ={}
                feature = features[f_cnt* args.predict_batch_size+i]
                l = len(feature.token_to_orig_map[each_ans['id']])
                s = min(l,int(each_ans['start'])+1)
                e = min(l,int(each_ans['end'])+2)
                real_start = feature.token_to_orig_map[each_ans['id']][s]
                real_end = feature.token_to_orig_map[each_ans['id']][e]
                temp_ans['question_type'] = p_example.question_type
                temp_ans['question'] = p_example.question_text
                temp_ans['question_id'] = p_example.qas_id
                temp_ans['answers'] = [''.join(p_example.documents[each_ans['id']]['doc_tokens'][real_start:real_end])]
                temp_ans['source'] = 'search'
                temp_ans['score'] =each_ans['score']
                 
                ans.append(temp_ans)
        f_cnt += 1
    # all_predictions, all_nbest_json = convert_output(predict_examples, features, all_results,
    #                                                 args.n_best_size, args.max_answer_length,
    #                                                 args.do_lower_case, args.verbose_logging)
    
    return ans


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    input_file = args.predict_file if evaluate else args.train_file
    cached_features_file = os.path.join(os.path.dirname(input_file), 'cached_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache and not output_examples:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", input_file)
        examples = read_baidu_examples(input_file=input_file, is_training=not evaluate)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                doc_stride=args.doc_stride,
                                                max_query_length=args.max_query_length,
                                                is_training=not evaluate)

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_q_input_ids = torch.tensor([f.q_input_ids for f in features], dtype=torch.long)
    all_q_input_mask = torch.tensor([f.q_input_mask for f in features], dtype=torch.long)
    all_q_segment_ids = torch.tensor([f.q_segment_ids for f in features], dtype=torch.long)
    # for f in features:
    #     if(len(f.p_input_ids)!=5):
    #         print(f)
    all_p_input_ids = torch.tensor([f.p_input_ids for f in features], dtype=torch.long)
    all_p_input_mask = torch.tensor([f.p_input_mask for f in features], dtype=torch.long)
    all_p_segment_ids = torch.tensor([f.p_segment_ids for f in features], dtype=torch.long)
    
    if evaluate:
        all_example_index = torch.arange(all_p_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_q_input_ids, all_q_input_mask, all_q_segment_ids,all_p_input_ids, 
                                    all_p_input_mask, all_p_segment_ids, all_example_index)
    else:
        all_right_num = torch.tensor([f.right_num for f in features], dtype=torch.int)
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        dataset = TensorDataset(all_q_input_ids, all_q_input_mask, all_q_segment_ids,
                                all_p_input_ids, all_p_input_mask, all_p_segment_ids,
                                all_start_positions, all_end_positions,all_right_num)
        
    if output_examples:
        return dataset, examples, features
    return dataset


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--train_file", default='./test.txt', type=str, required=True,
                        help="Duqa json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default=None, type=str, required=True,
                        help="Duqa json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-3, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        if(device == 'cpu'): #zhq: 设置CPU不用分布，测试专用
            args.n_gpu = 1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',rank= args.local_rank)
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    # model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    model = BertForBaiduQA_Answer_Selection(config=config)
    model.load_state_dict(state_dict= torch.load(args.model_name_or_path+'/pytorch_model.bin',map_location=lambda storage, loc: storage))
    for name, param in model.named_parameters():
        if('bert2.bert' in name):
	        param.requires_grad=False
        print(name,param.requires_grad)
    # for name, param in model.named_parameters():
	#     print(name,param.requires_grad)
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    # model = torch.nn.DataParallel(model)
    model.to(args.device)  # change this
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)

            # Evaluate
            evaluate(args, model, tokenizer, prefix=global_step)

    return 0
 
