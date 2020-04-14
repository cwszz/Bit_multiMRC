# -*- coding: utf-8 -*-
""" Load Duqa dataset. """

from __future__ import absolute_import, division, print_function

import collections
import json
import logging
import math
from io import open
from tqdm import tqdm

from transformers.tokenization_bert import BasicTokenizer, whitespace_tokenize

logger = logging.getLogger(__name__)


class BaiduExample(object):
    """
    A single training/test example for the baidu dataset
    包括ID， 问题，分词过的文本，fake answer， SP,EP
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 documents,
                 right_num = None,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,):
        self.qas_id = qas_id
        self.question_text = question_text
        self.documents = documents
        self.right_num = right_num
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", documetns: [%s]" % (" ".join(self.documents))
        if self.right_num:
            s += ", right_num: %d" % (self.right_num)
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        return s


class InputFeatures(object):
    """A single set of features of data."""
    # zhq: 增加问题的编码，目前觉得q不需要对照表所以没有map
    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 q_input_ids,
                 q_input_mask,
                 q_segment_ids,
                 p_input_ids,
                 p_input_mask,
                 p_segment_ids,
                 right_num=None,
                 start_position=None,
                 end_position=None):
        self.right_num = right_num
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.q_input_ids = q_input_ids
        self.q_input_mask = q_input_mask
        self.q_segment_ids = q_segment_ids
        self.p_input_ids = p_input_ids
        self.p_input_mask = p_input_mask
        self.p_segment_ids = p_segment_ids
        self.start_position = start_position
        self.end_position = end_position


def read_baidu_examples(input_file, is_training):
    """Read a baidu json file into a list of BaiduExample."""
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False
    flag = 0
    with open(input_file, "r", encoding='utf-8') as reader:
        examples = []
        for line in tqdm(reader, desc='reading baidu examples...'):
            example = json.loads(line)
            qas_id = example['question_id']
            question_text = example['question']
            right_num = example['right_doc']
            docs = example['documents']
            right_doc_tokens = example['documents'][right_num]['doc_tokens']
            # context_tokens = example['doc_tokens']
            start_position = None
            end_position = None
            orig_answer_text = None
            #若不是训练，那么数据应该只包含问题，文本，以上三个信息都为None
            #若是训练的话，
            if is_training:
                orig_answer_text = example['answer'][0]
                start_position = int(example['answer_span'][0])
                end_position = int(example['answer_span'][1])

                # 检测一下给出的fake answer 能否在文中找出来。 找不出来就跳过。
                actual_text = "".join(right_doc_tokens[start_position:(end_position+1)])
                cleaned_answer_text = orig_answer_text
                if actual_text.find(cleaned_answer_text) == -1:
                    flag += 1
                    continue
            per_example = BaiduExample(
                qas_id=qas_id,
                question_text=question_text,
                documents=docs,
                right_num = right_num,
                orig_answer_text=orig_answer_text,
                start_position=start_position,
                end_position=end_position,
                )
            examples.append(per_example)
    logger.warning("Could not find answer {}".format(flag))
    return examples


def read_baidu_examples_pred(raw_data, is_training):# 有个问题，dureader是多文档的，但是这个好像只有一个doc的文档。感觉mrc模型就是单文档模型
    """直接从[dir, dir...]读取数据"""

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for example in raw_data:
        # seg para就是 分词后的文本
        qas_id = example['question_id']
        question_text = example['question']
        # context_tokens = example['doc_tokens']
        right_num = example['right_doc']
        docs = example['documents']
        right_doc_tokens = example['documents'][right_num]['doc_tokens']
        # context_tokens = example['doc_tokens']
        start_position = None
        end_position = None
        orig_answer_text = None
        # 若不是训练，那么数据应该只包含问题，文本，以上三个信息都为None
        # 若是训练的话，
        if is_training:
            orig_answer_text = example['answer'][0]
            start_position = int(example['answer_span'][0])
            end_position = int(example['answer_span'][1])

            # 检测一下给出的fake answer 能否在文中找出来。 找不出来就跳过。
            actual_text = "".join(
                right_doc_tokens[start_position:(end_position + 1)])
            cleaned_answer_text = orig_answer_text
            if actual_text.find(cleaned_answer_text) == -1:
                logger.warning("Could not find answer: '%s' vs. '%s'",
                               actual_text, cleaned_answer_text)
                continue
        per_example = BaiduExample(
            qas_id=qas_id,
            question_text=question_text,
            documents=docs,
            right_num = right_num,
            orig_answer_text=orig_answer_text,
            start_position=start_position,
            end_position=end_position,
        )
        examples.append(per_example)
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    features = []
    for (example_index, example) in tqdm(enumerate(examples), desc='converting features...'):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]
        """问题的特征获取"""
        q_ids = tokenizer.convert_tokens_to_ids(query_tokens)
        q_input_ids = tokenizer.build_inputs_with_special_tokens(q_ids)
        q_segment_ids = tokenizer.create_token_type_ids_from_sequences(q_ids)
        q_input_mask = [1] * len(q_input_ids)
        while len(q_input_ids) < max_query_length:
            q_input_ids.append(0)
            q_input_mask.append(0)
            q_segment_ids.append(0)
        assert len(q_input_ids) == max_query_length
        assert len(q_input_mask) == max_query_length
        assert len(q_segment_ids) == max_query_length 
        """针对每一篇文章来获取文章特征""" 
        docs_tok_to_orig_index = []
        docs_start_position = []
        docs_end_position = []
        docs_orig_to_tok_index = []
        docs_all_doc_tokens = []
        docs_p_input_ids = []
        docs_p_input_masks = []
        docs_p_to_ori_map = []
        docs_p_tokens = []
        docs_p_segment_ids = [] #检查对应性
        for j in range(len(example.documents)):
            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(example.documents[j]['doc_tokens']):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = tokenizer.tokenize(token)           # 一个中文单词 eg:保存
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)
            docs_all_doc_tokens.append(all_doc_tokens)
            docs_orig_to_tok_index.append(orig_to_tok_index)
            docs_tok_to_orig_index.append(tok_to_orig_index)

            tok_start_position = None
            tok_end_position = None
            if j == example.right_num and  is_training:
                tok_start_position = orig_to_tok_index[example.start_position]
                if example.end_position < len(example.documents[example.right_num]['doc_tokens']) - 1:
                    tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1
                (tok_start_position, tok_end_position) = _improve_answer_span(
                    all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                    example.orig_answer_text)

            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

            # We can have documents that are longer than the maximum sequence length.
            # To deal with this we do a sliding window approach, where we take chunks
            # of the up to our max length with a stride of `doc_stride`.
            _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name  这里要用滑块，比如600长的文章分为0-500+，128-600+两篇文章
                "DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, doc_stride)

            for (doc_span_index, doc_span) in enumerate(doc_spans):
                tokens = []
                token_to_orig_map = {}
                token_is_max_context = {}
                p_segment_ids = []
                # 文章因为需要建立一个从字到词的对应表，所以还得一个一个弄
                tokens.append("[CLS]")
                p_segment_ids.append(0)

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                    is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                            split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    p_segment_ids.append(0)
                tokens.append("[SEP]")
                p_segment_ids.append(0)
                # cls = [tokenizer.cls_token_id]
                # p_segment_ids = tokenizer.create_token_type_ids_from_sequences(tokens)
                p_input_ids = tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                p_input_mask = [1] * len(p_input_ids)

                # Zero-pad up to the sequence length.
                
                while len(p_input_ids) < max_seq_length:
                    p_input_ids.append(0)
                    p_input_mask.append(0)
                    p_segment_ids.append(0)
                    
                assert len(p_input_ids) == max_seq_length
                assert len(p_input_mask) == max_seq_length
                assert len(p_segment_ids) == max_seq_length
                
                start_position = 0
                end_position = 0 

                if(j == example.right_num and is_training):
                    # For training, if our document chunk does not contain an annotation
                    # we throw it out, since there is nothing to predict.
                    doc_start = doc_span.start
                    doc_end = doc_span.start + doc_span.length - 1
                    if (example.start_position < doc_start or
                            example.end_position < doc_start or
                            example.start_position > doc_end or example.end_position > doc_end): #滑块中不包含答案
                        continue
                    # doc_offset = len(query_tokens) + 2 因为是删掉了query所以没有offset
                    start_position = tok_start_position - doc_start 
                    end_position = tok_end_position - doc_start
        
                docs_p_input_ids.append(p_input_ids)
                docs_p_input_masks.append(p_input_mask)
                docs_p_segment_ids.append(p_segment_ids)
                docs_p_to_ori_map.append(token_to_orig_map)
                docs_p_tokens.append(tokens)
                docs_start_position.append(start_position)
                docs_end_position.append(end_position)

                break # 如果是一个或多个滑块且是无答案的文档直接取第一个，如果是一个或多个滑块包含答案的文档，直接取有答案的第一个滑块
        features.append(
            InputFeatures(
                right_num = example.right_num,
                unique_id=unique_id,
                example_index=example_index,
                doc_span_index=doc_span_index, #应该无用了，start在上面减掉了offset
                tokens=docs_p_tokens,
                token_to_orig_map=docs_p_to_ori_map,
                token_is_max_context=token_is_max_context, # 目前没看懂有没有用
                q_input_ids=q_input_ids,
                q_input_mask=q_input_mask,
                q_segment_ids=q_segment_ids,
                p_input_ids=docs_p_input_ids,
                p_input_mask=docs_p_input_masks,
                p_segment_ids=docs_p_segment_ids,
                start_position=docs_start_position,
                end_position=docs_end_position))
        unique_id += 1

    return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    #因为word piece tokenizer，要进一步拆词，对应的answer span 需要改变。
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, verbose_logging):
    """Write final predictions to the json file."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True
            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1
        i = 0

        all_predictions[example.qas_id] = nbest_json[0]["text"]
        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4,ensure_ascii=False) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4,ensure_ascii=False) + "\n")


def convert_output(all_examples, all_features, all_results, n_best_size,
                    max_answer_length, do_lower_case, verbose_logging):
    """TBD"""
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit", "start_prob", "end_prob", "start_prob_v1", "end_prob_v1"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                            start_prob=_compute_softmax([result.start_logits[start_index], result.end_logits[start_index]])[0],
                            start_prob_v1=_compute_sigmoid(result.start_logits[start_index]),
                            end_prob=_compute_softmax([result.start_logits[end_index], result.end_logits[end_index]])[1],
                            end_prob_v1=_compute_sigmoid(result.end_logits[end_index])))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["raw_text","text", "start_logit", "end_logit", "start_prob", "end_prob", "start_prob_v1", "end_prob_v1","start_index","end_index"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True
            nbest.append(
                _NbestPrediction(
                    raw_text = ''.join(feature.tokens),
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                    start_prob=pred.start_prob,
                    start_prob_v1=pred.start_prob_v1,
                    end_prob=pred.end_prob,
                    end_prob_v1=pred.end_prob_v1,
                    start_index = pred.start_index,
                    end_index = pred.end_index))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, start_prob=0.0, start_prob_v1=0.0, end_prob=0.0, end_prob_v1=0.0))

        assert len(nbest) >= 1

        total_scores = []
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["raw_text"] = entry.raw_text
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            output["start_prob"] = entry.start_prob
            output["start_prob_v1"] = entry.start_prob_v1
            output["end_prob"] = entry.end_prob
            output["end_prob_v1"] = entry.end_prob_v1
            output["start_index"] = entry.start_index
            output["end_index"]= entry.end_index

            nbest_json.append(output)

        assert len(nbest_json) >= 1
        i = 0
        all_predictions[example.qas_id] = nbest_json[0]["text"]
        all_nbest_json[example.qas_id] = nbest_json

    return all_predictions, all_nbest_json


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

def _compute_sigmoid(score):
    return 1/(1 + math.exp(-score))
