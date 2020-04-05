# The following classes are added by BIT_OpenDomain_QA team
# @Time : 2019-09-30 13:48
# @Author : Mucheng Ren, Ran Wei, Hongyu Liu, Yu Bai, Yang Wang
# @Email : rdoctmc@gmail.com, weiranbit@163.com, liuhongyu12138@gmail.com, Wnwhiteby@gamil.com, wangyangbit1@gmail.com

import torch
from torch import nn
from torch.nn import functional as F
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
        """Q-P match exactly Bi-attention"""
        self.lstm = nn.LSTM(input_size=config.hidden_size,hidden_size=config.hidden_size,
            num_layers=self.lstmlayers,bidirectional=True,batch_first=False)
        self.score = nn.Linear(self.lstmlayers* 2 * 3 * config.hidden_size, 1)
        # 2是双向，3是Bi-DAF （p,q,p·q）元素对应相乘
        self.lstm_m = nn.LSTM(input_size=config.hidden_size*8, hidden_size=config.hidden_size*8 ,num_layers=1,
            bidirectional=True,batch_first=True)
        # =======zhq
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.init_weights()
    
    def forward(self, q_input_ids,  p_input_ids,q_attention_mask=None, q_token_type_ids=None, q_position_ids=None, q_head_mask=None,
                p_attention_mask=None, p_token_type_ids=None, p_position_ids=None, p_head_mask=None,
                start_positions=None, end_positions=None):
        """Embedding"""
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
        """Feature_Extraction"""
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
        p_features,(p_final_hidden_state, p_final_cell_state) = self.lstm(p_embedding,(h_0_p,c_0_p)) # seq_length batch hidden
        q_features,(q_final_hidden_state, q_final_cell_state) = self.lstm(q_embedding,(h_0_q,c_0_q))
        """Bi-Attention"""
        u = torch.zeros(p_features.size(0),q_features.size(0),q_features.size(1))
        # q_feature.size(1)其实就是batchsize
        for t,p_word in enumerate(p_features):
            for j,q_word in enumerate(q_features):
                u[t][j] = self.score(torch.cat((p_word,q_word,q_word.mul(p_word)),1)).squeeze(1)  
                #从第一维来合并，输入到线性层里。线性层的输入为batch * hidden_size
        q2c_attention = F.softmax(u,1) # dim=1 表示行加和为1
            # U_t = 对j叠加，a_(t,j)*U_j 所以j也就是query_length加和为1
        c2q_attention = torch.max(u,1)[0] # dim = 0 表示从512个第0维中找出代表512个最大的
        q2c_attention = q2c_attention.transpose(0,2).transpose(1,2)
        c2q_attention = c2q_attention.transpose(0,1) 
        q_features = q_features.transpose(0,1) #变成 22，64，1536
        p_features = p_features.transpose(0,1) #变成 22, 512，1536
        new_p_features_u = torch.zeros(q_features.size(0),q2c_attention.size(1),q_features.size(-1)) 
            # 22, 512, 1536
        # 构建加权
        for i,(each_attn,each_q_word) in enumerate(zip(q2c_attention,q_features)):
            new_p_features_u[i] = each_attn.mm(each_q_word)
        # 获得q2c的加权特征
        new_p_features_h = torch.zeros(p_features.size())
        for i, (each_c2q,p_feature) in enumerate(zip(c2q_attention,p_features)):
            for j,(c2q_weight,p_word_feature) in enumerate(zip(each_c2q,p_feature)):
                new_p_features_h[i][j] = c2q_weight * p_word_feature
        final_p_features = torch.zeros(p_features.size(0),p_features.size(1),p_features.size(2)*4)    
        # final_p_features = torch.cat((p_features,new_p_features_h,p_features.mul(new_p_features_h),p_features.mul(new_p_features_u)),-1)  
        # for i, (each_p,each_h,each_u) in enumerate(zip(p_features,new_p_features_h,new_p_features_u)):
        #     for j, (each_p_word,each_h_word,each_u_word) in enumerate(zip(each_p,each_h,each_u)):
        #         final_p_features[i][j] = torch.cat((each_p_word,each_u_word,
        #             each_p_word.mul(each_u_word),each_p_word.mul(each_h_word)),0)
        
        # final_p_features2 = torch.zeros(p_features.size(0),p_features.size(1),p_features.size(2)*4)
        final_p_features = torch.cat((p_features,new_p_features_h,p_features.mul(new_p_features_u),p_features.mul(new_p_features_h)),-1)  
        # 构建最终的表示 [h;u ̃;h◦u ̃;h◦h ̃]
        final_p_features =final_p_features.transpose(0,1)
        h_0_p = Variable(torch.zeros(self.lstmlayers*2, final_p_features.size(0), self.config.hidden_size*8)) # 乘2因为是双向
        c_0_p = Variable(torch.zeros(self.lstmlayers*2, final_p_features.size(0), self.config.hidden_size*8))
        final_p_features,(p_final_hidden_state, p_final_cell_state) = self.lstm_m(final_p_features,(h_0_p,c_0_p)) # seq_length batch hidden
        # new_p_features = torch.Tensor(new_p_features)  
        print(new_p_features_h.size())
        
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
