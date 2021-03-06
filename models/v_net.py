# The following classes are added by BIT_OpenDomain_QA team
# @Time : 2019-09-30 13:48
# @Author : Mucheng Ren, Ran Wei, Hongyu Liu, Yu Bai, Yang Wang
# @Email : rdoctmc@gmail.com, weiranbit@163.com, liuhongyu12138@gmail.com, Wnwhiteby@gamil.com, wangyangbit1@gmail.com
import os
import torch
# import inspect
# from gpu_mem_track import MemTracker
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from torch.autograd import Variable
from transformers import (BertConfig,BertModel,BertPreTrainedModel)
from transformers.file_utils import add_start_docstrings
from transformers.modeling_bert import (BERT_INPUTS_DOCSTRING,
                                        BERT_START_DOCSTRING, BertModel,
                                        BertPreTrainedModel)

# os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'

@add_start_docstrings("""Bert Model with a span classification head on top for extractive question-answering tasks like Duqa (a linear layers on top of
    the hidden-states output to compute `span start logits` and `span end logits`). """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForBaiduQA_Answer_Selection(BertPreTrainedModel):
# class BertForBaiduQA_Answer_Selection(AlbertPreTrainedModel):
    """ TBD """
    def __init__(self, config):
        super(BertForBaiduQA_Answer_Selection, self).__init__(config)
        self.temp_hidden = 230
        # self.bert =AlbertModel(config)
        self.bert = BertModel(config)
        self.lstmlayers = 1
        self.config = config
        # -----zhq
        """Q-P match exactly Bi-attention"""
        self.lstm = nn.LSTM(input_size=config.hidden_size,hidden_size=self.temp_hidden,
            num_layers=self.lstmlayers,bidirectional=True,batch_first=False)
        self.score = nn.Linear(self.lstmlayers* 2 * 3 * self.temp_hidden, 1)
        # 2是双向，3是Bi-DAF （p,q,p·q）元素对应相乘
        self.lstm_m = nn.LSTM(input_size=self.temp_hidden*8, hidden_size=self.temp_hidden ,num_layers=1,
            bidirectional=True,batch_first=True)
        self.w2_a = nn.Linear(self.temp_hidden * 2 * 2,self.temp_hidden *2)
        self.w1_a = nn.Linear(self.temp_hidden * 2, 1)
        self.lstm_boundary = nn.LSTM(input_size= self.temp_hidden* 2, hidden_size=self.temp_hidden,num_layers=1,
            bidirectional=True,batch_first= True) 
        """Answer_content predict"""
        self.content_predict = nn.Sequential(
            nn.Linear(self.temp_hidden*2,self.temp_hidden),
            nn.ReLU(inplace=False),
            nn.Linear(self.temp_hidden,1),
            nn.Sigmoid()
        )
        self.w3_a = nn.Linear(config.hidden_size * 3, 1)
        # self.w2_c = nn.Linear(config.hidden_size * 2,config.hidden_size)
        # self.w1_c = nn.Linear(config.hidden_size, 1)
        # =======zhq
        # self.qa_outputs = nn.Linear(self.temp_hidden, 2)
        self.init_weights()
    
    def forward(self, q_input_ids,  p_input_ids,q_attention_mask=None, q_token_type_ids=None, q_position_ids=None, q_head_mask=None,
                p_attention_mask=None, p_token_type_ids=None, p_position_ids=None, p_head_mask=None,right_num = None,
                start_positions=None, end_positions=None):
        # frame = inspect.currentframe() 
        # gpu_tracker = MemTracker(frame) 
        device = p_input_ids.device
        """transpose batch and docs"""
        # gpu_tracker.track()
        p_input_ids = p_input_ids.transpose(0,1)
        p_attention_mask = p_attention_mask.transpose(0,1)
        p_token_type_ids = p_token_type_ids.transpose(0,1)
        """Embedding"""
        # gpu_tracker.track()
        q_outputs = self.bert(q_input_ids,
                            attention_mask=q_attention_mask,
                            token_type_ids=q_token_type_ids,
                            position_ids=q_position_ids, 
                            head_mask=q_head_mask)
        q_embedding = q_outputs[0] # size(batch,seq,hidden)
        q_embedding = q_embedding.transpose(0,1)
        # gpu_tracker.track()
        # R_A是 docs batch hidden
        # test_loss = torch.zeros(2,1).to(device)
        final_loss = torch.zeros(q_embedding.size(1),1).to(device)
        Doc_ra = torch.zeros(p_input_ids.size(0),p_input_ids.size(1),q_embedding.size(-1)).to(device) # q_embedding是hidde_size
        if(start_positions is not None and end_positions is not None):
            start_positions = start_positions.transpose(0,1)
            end_positions = end_positions.transpose(0,1)
        for doc_index, (each_doc_p_input_id,each_doc_p_attention_mask,each_doc_p_token_type) in enumerate(zip(p_input_ids,p_attention_mask, p_token_type_ids)):
            # gpu_tracker.track()
            p_outputs = self.bert(each_doc_p_input_id,
                                attention_mask=each_doc_p_attention_mask,
                                token_type_ids=each_doc_p_token_type,
                                position_ids=p_position_ids, 
                                head_mask=p_head_mask)
            p_embedding = p_outputs[0]
            p_embedding = p_embedding.transpose(0,1)
            """Feature_Extraction"""
            if(device == "cpu"):
                h_0_q = Variable(torch.zeros(self.lstmlayers*2, q_input_ids.size(0), self.temp_hidden)) # 乘2因为是双向
                c_0_q = Variable(torch.zeros(self.lstmlayers*2, q_input_ids.size(0), self.temp_hidden))
                h_0_p = Variable(torch.zeros(self.lstmlayers*2, p_input_ids.size(1), self.temp_hidden)) # 乘2因为是双向 p_input_ids 为 docs batchs seq hidden
                c_0_p = Variable(torch.zeros(self.lstmlayers*2, p_input_ids.size(1), self.temp_hidden))
                h_0_p_bound = Variable(torch.zeros(self.lstmlayers*2, p_input_ids.size(1), self.temp_hidden)) # 乘2因为是双向
                c_0_p_bound = Variable(torch.zeros(self.lstmlayers*2, p_input_ids.size(1), self.temp_hidden))
            else:
                h_0_q = Variable(torch.zeros(self.lstmlayers*2, q_input_ids.size(0), self.temp_hidden)).to(device) # 乘2因为是双向
                c_0_q = Variable(torch.zeros(self.lstmlayers*2, q_input_ids.size(0), self.temp_hidden)).to(device)
                h_0_p = Variable(torch.zeros(self.lstmlayers*2, p_input_ids.size(1), self.temp_hidden)).to(device) # 乘2因为是双向
                c_0_p = Variable(torch.zeros(self.lstmlayers*2, p_input_ids.size(1), self.temp_hidden)).to(device)
                h_0_p_bound = Variable(torch.zeros(self.lstmlayers*2, p_input_ids.size(1), self.temp_hidden)).to(device) # 乘2因为是双向
                c_0_p_bound = Variable(torch.zeros(self.lstmlayers*2, p_input_ids.size(1), self.temp_hidden)).to(device)
            p_features,(p_final_hidden_state, p_final_cell_state) = self.lstm(p_embedding,(h_0_p,c_0_p)) # seq_length batch hidden
            q_features,(q_final_hidden_state, q_final_cell_state) = self.lstm(q_embedding,(h_0_q,c_0_q))
            """Bi-Attention"""
            u = torch.zeros(p_features.size(0),q_features.size(0),q_features.size(1)).to(device)
            # q_feature.size(1)其实就是batchsize
            for t,p_word in enumerate(p_features):
                for j,q_word in enumerate(q_features):
                    u[t][j] = self.score(torch.cat((p_word,q_word,q_word.mul(p_word)),1)).squeeze(1)  
                    #从第一维来合并，输入到线性层里。线性层的输入为batch * hidden_size
            """构造最终p的表示"""
            q2c_attention = F.softmax(u,1) # dim=1 表示行加和为1
                # U_t = 对j叠加，a_(t,j)*U_j 所以j也就是query_length加和为1
            c2q_attention = torch.max(u,1)[0] # dim = 0 表示从512个第0维中找出代表512个最大的
            q2c_attention = q2c_attention.transpose(0,2).transpose(1,2)
            c2q_attention = c2q_attention.transpose(0,1) 
            q_features = q_features.transpose(0,1) #变成 22，64，1536
            p_features = p_features.transpose(0,1) #变成 22, 512，1536
            # new_p_features_u = torch.zeros(q_features.size(0),q2c_attention.size(1),q_features.size(-1)).to(device)
            #     # 22, 512, 1536
            # # 构建加权
            # for i,(each_attn,each_q_word) in enumerate(zip(q2c_attention,q_features)):
            #     new_p_features_u[i] = each_attn.mm(each_q_word)
            new_p_features_u = q2c_attention.bmm(q_features)
            # 获得q2c的加权特征  可能可以简化一下
            new_p_features_h = torch.zeros(p_features.size()).to(device)
            for i, (each_c2q,p_feature) in enumerate(zip(c2q_attention,p_features)):
                for j,(c2q_weight,p_word_feature) in enumerate(zip(each_c2q,p_feature)):
                    new_p_features_h[i][j] = c2q_weight * p_word_feature
            final_p_features = torch.zeros(p_features.size(0),p_features.size(1),p_features.size(2)*4).to(device)    
            final_p_features = torch.cat((p_features,new_p_features_h,p_features.mul(new_p_features_u),p_features.mul(new_p_features_h)),-1)  
            # 构建最终的表示 [h;u ̃;h◦u ̃;h◦h ̃]
            # final_p_features =final_p_features.transpose(0,1) 已经是batch first 不用转了
            # h_0_p = Variable(torch.zeros(self.lstmlayers*2, final_p_features.size(0), self.temp_hidden)).to(device) # 乘2因为是双向
            # c_0_p = Variable(torch.zeros(self.lstmlayers*2, final_p_features.size(0), self.temp_hidden)).to(device)
            # final_p_features,(p_final_hidden_state, p_final_cell_state) = self.lstm_m(final_p_features,(h_0_p,c_0_p)) # seq_length batch hidden
            final_p_features = self.lstm_m(final_p_features,(h_0_p,c_0_p))[0] # seq_length batch hidden
            # 再经过一次LSTM
            # Pointer Network
            # final_p_features = final_p_features.transpose(0,1) # 变成batch seq hid
            """第一次计算START，alpha1就是开始概率"""
            h_0_a = Variable(torch.zeros(final_p_features.size(0),1,final_p_features.size(2))).to(device) # 这个不知道人家咋初始化的
            # g_1 = self.w1_a(torch.tanh(self.w2_a(torch.cat((final_p_features,h_0_a.repeat(1,final_p_features.size(1),1))
            #     ,2)))).squeeze(2) # 把h_0_a 扩充成512个词的，在hidden层拼接
            # g_1 是 公式（5）
            alpha_1 = torch.softmax(self.w1_a(torch.tanh(self.w2_a(torch.cat((final_p_features,h_0_a.repeat(1,final_p_features.size(1),1))
                ,2)))).squeeze(2),1).unsqueeze(1) #(6)
            # c_1 = torch.zeros(final_p_features.size(0),1, final_p_features.size(2)).to(device) #hiddensize 和h_0_a一致
            # for i, (each_alpha_1,each_p_feature) in enumerate(zip(alpha_1,final_p_features)):
            #     c_1[i] = each_alpha_1.mm(each_p_feature)
            c_1 = alpha_1.bmm(final_p_features)
            h_0_a = self.lstm_boundary(c_1,(h_0_p_bound,c_0_p_bound))[0]
            """第二次计算END，alpha2就是结束概率"""
            # g_2 = self.w1_a(torch.tanh(self.w2_a(torch.cat((final_p_features,h_0_a.repeat(1,final_p_features.size(1),1))
            #     ,2)))).squeeze(2) # 把h_0_a 扩充成512个词的，在hidden层拼接
            alpha_2 = torch.log(torch.softmax(self.w1_a(torch.tanh(self.w2_a(torch.cat((final_p_features,h_0_a.repeat(1,final_p_features.size(1),1))
                ,2)))).squeeze(2),1))
            alpha_1 =  torch.log(alpha_1.squeeze(1))
            # alpha_2 = torch.log(alpha_2)
            # alpha_1 =alpha_1)
            ans_total_loss = torch.zeros(q_embedding.size(1)).to(device)
            """计算ground_truth的概率"""
            # poss = torch.sigmoid(self.w1_c(torch.relu(self.w2_c(final_p_features))))
            poss = self.content_predict(final_p_features)
            """Verify ready_Part"""
            # verify_poss = poss.transpose(1,2)
            # r_A = torch.zeros(final_p_features.size(0),p_embedding.size(-1))
            p_embedding = p_embedding.transpose(0,1)
            # for word_num, (each_poss,each_p_embedding)  in enumerate(zip(poss.transpose(1,2),p_embedding)):
                # Doc_ra[doc_index][word_num] = torch.div(each_poss.mm(each_p_embedding),p_embedding.size(1))
            Doc_ra[doc_index] = torch.div(poss.transpose(1,2).bmm(p_embedding).squeeze(1),p_embedding.size(1))
            # Doc_ra[doc_index] = r_A
            # gpu_tracker.track()
            """"针对包含答案的文章，计算Loss""" # not None 是为了判断是否是Train
            if start_positions is not None and end_positions is not None :
                for t in range(start_positions.size(1)):  # 每个batch
                    if(end_positions[doc_index][t] == 0):
                        continue
                    else:
                        """part 1 loss"""
                        # if len(start_positions[doc_index].size()) > 1:
                        #     start_positions[doc_index] = start_positions[doc_index].squeeze(-1)
                        # if len(end_positions[doc_index].size()) > 1:
                        #     end_positions[doc_index] = end_positions[doc_index].squeeze(-1)
                        # sometimes the start/end positions are outside our model inputs, we ignore these terms
                        ignored_index = alpha_1.size(1)
                        
                        temp_start = torch.clamp(start_positions[doc_index][t],0, ignored_index) #控制范围
                        temp_end = torch.clamp(end_positions[doc_index][t],0, ignored_index) 
                        loss_func = nn.NLLLoss(ignore_index= ignored_index)
                      
                        # start_loss = loss_func(alpha_1[t].unsqueeze(0),start_positions[doc_index][t].reshape(1))
                        # end_loss = loss_func(alpha_2[t].unsqueeze(0), end_positions[doc_index][t].reshape(1))
                        ans_total_loss[t] = (loss_func(alpha_1[t].unsqueeze(0),temp_start.reshape(1))
                                    + loss_func(alpha_2[t].unsqueeze(0), temp_end.reshape(1)) )/2  # 这个loss算的没毛病，pointer标准的loss  
                        """part 2 loss"""
                        # neg_poss = torch.ones(poss.size()).to(device).add(poss * -1)  # 1 - poss
                        answer_span = torch.zeros(poss.size(0),poss.size(1)) # batch * words
                        # no_answer_span = torch.ones(poss.size(0),poss.size(1))
                        for (each_answer_span, start_position,end_position) in zip(answer_span,start_positions[doc_index],end_positions[doc_index]):
                            for i in range(end_position - start_position):
                                each_answer_span[i+start_position] = 1
                                # each_no_answer_span[i+start_position] = 0
                        content_loss = torch.zeros(1).to(device)
                        for (each_batch_answer_span,each_batch_poss) in zip(answer_span,poss):
                            for (each_word_ans,each_word_poss) in zip(each_batch_answer_span,each_batch_poss):
                                content_loss[0] = content_loss[0]+ each_word_ans * torch.log(each_word_poss) + (1-each_word_ans) * torch.log(1-each_word_poss)
                        content_loss[0] = -1 * content_loss[0]/(poss.size(0)*poss.size(1))+ans_total_loss[0]
                        # return content_loss
                        final_loss[t] =  ans_total_loss[0] + 0.5 * content_loss[0] 
            # torch.cuda.empty_cache() 
        # s = torch.zeros(p_input_ids.size(1), p_input_ids.size(0),p_input_ids.size(0)).to(device) # batch docs docs 因为下面好乘
        # for i , each_ra in enumerate(Doc_ra.transpose(0,1)):
        #     s[i]= each_ra.mm(each_ra.T)  # 应该是 batch docs docs
        #     for j in range(s[i].size(1)):
        #         s[i][j][j] = 0
        s = Doc_ra.transpose(0,1).bmm(Doc_ra.transpose(0,1).transpose(1,2))
        for i in range(s.size(0)):
            for j in range(s.size(1)):
                s[i][j][j] = 0
        
        s = torch.softmax(s,-1)
        # verify_rpt = torch.zeros(p_input_ids.size(1),p_input_ids.size(0),Doc_ra.size(-1)).to(device) # batch docs hidden
        # for i, (each_s,each_ra) in enumerate(zip(s,Doc_ra.transpose(0,1))):
        #     verify_rpt[i] = each_s.mm(each_ra)
        verify_rpt = s.bmm(Doc_ra.transpose(0,1))
        p = torch.zeros(p_input_ids.size(0),p_input_ids.size(1)).to(device)
        p = torch.softmax(self.w3_a(torch.cat((Doc_ra.transpose(0,1),verify_rpt,verify_rpt.mul(Doc_ra.transpose(0,1))),-1)),1)
        p = torch.log(p).squeeze(-1)
        if  start_positions is not None and end_positions is not None:
            sum_loss = final_loss.transpose(0,1).mm(torch.ones(p.size(0),1).to(device))
            loss_func = nn.NLLLoss(ignore_index= 512)
            sum_loss = sum_loss + 0.5 * loss_func(p,right_num.long())
            return sum_loss
        else: # 预测时：
            return p
        # return (ans_total_loss + 0.5 * content_loss)
            # outputs = (total_loss,) + (alpha_1,alpha_2)
      

        # return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
