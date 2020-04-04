# The following classes are added by BIT_OpenDomain_QA team
# @Time : 2019-09-30 13:48
# @Author : Mucheng Ren, Ran Wei, Hongyu Liu, Yu Bai, Yang Wang
# @Email : rdoctmc@gmail.com, weiranbit@163.com, liuhongyu12138@gmail.com, Wnwhiteby@gamil.com, wangyangbit1@gmail.com

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from torch.autograd import Variable
from transformers.file_utils import add_start_docstrings
from transformers.modeling_bert import (BERT_INPUTS_DOCSTRING,
                                        BERT_START_DOCSTRING, BertModel,
                                        BertPreTrainedModel)

@add_start_docstrings("""Bert Model with a span classification head on top for extractive question-answering tasks like Duqa (a linear layers on top of
    the hidden-states output to compute `span start logits` and `span end logits`). """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForBaiduQA_Answer_Selection(BertPreTrainedModel):
    """ TBD """
    def __init__(self, config):
        super(BertForBaiduQA_Answer_Selection, self).__init__(config)
        self.bert = BertModel(config)
        self.lstmlayers = 1
        self.config = config
        # -----zhq
        self.lstm = nn.LSTM(input_size=config.hidden_size,hidden_size=config.hidden_size,
            num_layers=self.lstmlayers,bidirectional=True,batch_first=False)
        # =======zhq
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.init_weights()
    
    def forward(self, q_input_ids,  p_input_ids,q_attention_mask=None, q_token_type_ids=None, q_position_ids=None, q_head_mask=None,
                p_attention_mask=None, p_token_type_ids=None, p_position_ids=None, p_head_mask=None,
                start_positions=None, end_positions=None):

        p_outputs = self.bert(p_input_ids,
                            attention_mask=p_attention_mask,
                            token_type_ids=p_token_type_ids,
                            position_ids=p_position_ids, 
                            head_mask=p_head_mask)
        q_outputs = self.bert(q_input_ids,
                            attention_mask=q_attention_mask,
                            token_type_ids=q_token_type_ids,
                            position_ids=q_position_ids, 
                            head_mask=q_head_mask)
        p_embedding = p_outputs[0]
        q_embedding = q_outputs[0] # size(batch,seq,hidden)
        p_embedding = p_embedding.transpose(0,1)
        q_embedding = q_embedding.transpose(0,1)
        # if self.use_cuda:
        #     h_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size).cuda())
        #     c_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size).cuda())
        # else:
        #     h_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size))
        #     c_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size))
        #     注意把h,c放到cuda上
        h_0_q = Variable(torch.zeros(self.lstmlayers*2, q_input_ids.size(0), self.config.hidden_size)) # 乘2因为是双向
        c_0_q = Variable(torch.zeros(self.lstmlayers*2, q_input_ids.size(0), self.config.hidden_size))
        h_0_p = Variable(torch.zeros(self.lstmlayers*2, p_input_ids.size(0), self.config.hidden_size)) # 乘2因为是双向
        c_0_p = Variable(torch.zeros(self.lstmlayers*2, p_input_ids.size(0), self.config.hidden_size))
        p_features,(p_final_hidden_state, p_final_cell_state) = self.lstm(p_embedding,(h_0_p,c_0_p))
        q_features,(q_final_hidden_state, q_final_cell_state) = self.lstm(q_embedding,(h_0_q,c_0_q))
        logits = self.qa_outputs(sequence_output)  # logits 是batch里所有的数据个数32 * 长度512 * (start and end) 2
        start_logits, end_logits = logits.split(1, dim=-1)       
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1) #提取成end_logits

        outputs = (start_logits, end_logits,) + p_outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2  # 这个loss算的没毛病，pointer标准的loss
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
