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

class Feature_Extraction(BertPreTrainedModel):
    
    def __init__(self,config):
        super(Feature_Extraction,self).__init__(config)
        self.temp_hidden = 200
        self.bert = BertModel(config)
        self.bert.train
        self.lstmlayers = 1
        self.config = config
        self.lstm = nn.LSTM(input_size=config.hidden_size,hidden_size=self.temp_hidden,
            num_layers=self.lstmlayers,bidirectional=True,batch_first=False)
        # self.init_weights()

    def forward(self,input_ids,attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        device = input_ids.device
        h_0_q = Variable(torch.zeros(self.lstmlayers*2, input_ids.size(0), self.temp_hidden)).to(device) # 乘2因为是双向
        c_0_q = Variable(torch.zeros(self.lstmlayers*2, input_ids.size(0), self.temp_hidden)).to(device)
        input_embedding = self.bert(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids, 
                                head_mask=head_mask)
        self.lstm.flatten_parameters()
        features = self.lstm(input_embedding[0].transpose(0,1),(h_0_q,c_0_q))[0]
        return features,input_embedding[0]


class Ptr_net(nn.Module):
    def __init__(self):
        super(Ptr_net,self).__init__()
        self.temp_hidden = 200
        self.lstmlayers = 1
        self.score = nn.Linear(self.lstmlayers* 2 * 3 * self.temp_hidden, 1)
        self.lstm_m = nn.LSTM(input_size=self.temp_hidden*8, hidden_size=self.temp_hidden ,num_layers=1,
            bidirectional=True,batch_first=True)
        self.w2_a = nn.Linear(self.temp_hidden * 2 * 2,self.temp_hidden *2)
        self.w1_a = nn.Linear(self.temp_hidden * 2, 1)
        self.lstm_boundary = nn.LSTM(input_size= self.temp_hidden* 2, hidden_size=self.temp_hidden,num_layers=1,
            bidirectional=True,batch_first= True) 
        # self.init_weights()
    
    def forward(self,p_features,q_features):
        device = p_features.device
        h_0_p = Variable(torch.zeros(self.lstmlayers*2, p_features.size(1), self.temp_hidden)).to(device) # 乘2因为是双向
        c_0_p = Variable(torch.zeros(self.lstmlayers*2, p_features.size(1), self.temp_hidden)).to(device)
        h_0_p_bound = Variable(torch.zeros(self.lstmlayers*2, p_features.size(1), self.temp_hidden)).to(device) # 乘2因为是双向
        c_0_p_bound = Variable(torch.zeros(self.lstmlayers*2, p_features.size(1), self.temp_hidden)).to(device)
        self.lstm_boundary.flatten_parameters()
        self.lstm_m.flatten_parameters()
        u = torch.zeros(p_features.size(1),p_features.size(0),q_features.size(0)).to(device)
        for i,(p_features_batch,q_features_batch) in enumerate(zip(p_features.transpose(0,1).transpose(1,2),q_features.transpose(0,1).transpose(1,2))):
            # sec_u = p_features_batch.unsqueeze(-1).bmm(q_features_batch.unsqueeze(1)).transpose(0,1).transpose(1,2)
            # ss = p_features_batch.transpose(0,1).unsqueeze(1).repeat(1,q_features.size(0),1)
            # cc = q_features_batch.transpose(0,1).unsqueeze(0).repeat(p_features.size(0),1,1)
            cat_feature = torch.cat((p_features_batch.transpose(0,1).unsqueeze(1).repeat(1,q_features.size(0),1),
                                    q_features_batch.transpose(0,1).unsqueeze(0).repeat(p_features.size(0),1,1),
                                    p_features_batch.unsqueeze(-1).bmm(q_features_batch.unsqueeze(1)).transpose(0,1).transpose(1,2)),-1)
            u[i] = self.score(cat_feature.reshape(cat_feature.size(0)*cat_feature.size(1),-1)).squeeze(-1).reshape(p_features.size(0),q_features.size(0))
        # for t,p_word in enumerate(p_features):
        #     for j,q_word in enumerate(q_features):
        #         u[t][j] = self.score(torch.cat((p_word,q_word,q_word.mul(p_word)),1)).squeeze(1) 
        u = u.transpose(0,1).transpose(1,2)
        q2c_attention = F.softmax(u,1).transpose(0,2).transpose(1,2) # dim=1 表示行加和为1
                # U_t = 对j叠加，a_(t,j)*U_j 所以j也就是query_length加和为1
        c2q_attention = torch.max(u,1)[0].transpose(0,1) # dim = 0 表示从512个第0维中找出代表512个最大的
        new_p_features_u = q2c_attention.bmm(q_features.transpose(0,1))
            # 获得q2c的加权特征  可能可以简化一下
        p_features = p_features.transpose(0,1)
        new_p_features_h = torch.zeros(p_features.size()).to(device)
        for i, (each_c2q,p_feature) in enumerate(zip(c2q_attention,p_features)):
            for j,(c2q_weight,p_word_feature) in enumerate(zip(each_c2q,p_feature)):
                    new_p_features_h[i][j] = c2q_weight * p_word_feature
        final_p_features = torch.cat((p_features,new_p_features_h,p_features.mul(new_p_features_u),p_features.mul(new_p_features_h)),-1)  
        final_p_features = self.lstm_m(final_p_features,(h_0_p,c_0_p))[0]
        h_0_a = Variable(torch.zeros(final_p_features.size(0),1,final_p_features.size(2))).to(device) # 这个不知道人家咋初始化的
        alpha_1 = F.softmax(self.w1_a(torch.tanh(self.w2_a(torch.cat((final_p_features,h_0_a.repeat(1,final_p_features.size(1),1))
            ,2)))).squeeze(2),1).unsqueeze(1) #(6)
        c_1 = alpha_1.bmm(final_p_features)
        h_0_a = self.lstm_boundary(c_1,(h_0_p_bound,c_0_p_bound))[0]
        """第二次计算END，alpha2就是结束概率"""
        alpha_2 = torch.log(F.softmax(self.w1_a(torch.tanh(self.w2_a(torch.cat((final_p_features,h_0_a.repeat(1,final_p_features.size(1),1))
            ,2)))).squeeze(2),1))
        # alpha_2 = torch.log(alpha_2)
        alpha_1 =  torch.log(alpha_1.squeeze(1))

        return final_p_features, alpha_1, alpha_2


class Content_detect(nn.Module):
    def __init__(self):
        super(Content_detect,self).__init__()
        self.temp_hidden =200
        self.content_predict = nn.Sequential(
            nn.Linear(self.temp_hidden*2,self.temp_hidden),
            nn.ReLU(inplace=False),
            nn.Linear(self.temp_hidden,1),
            nn.Sigmoid()
        )
    def forward(self,final_features,p_embedding):
        poss = self.content_predict(final_features)
        representation = torch.div(poss.transpose(1,2).bmm(p_embedding).squeeze(1),p_embedding.size(1))
        return poss, representation

class Verify_ans(nn.Module):
    def __init__(self,config):
        super(Verify_ans,self).__init__()
        self.w3_a = nn.Linear(config.hidden_size * 3, 1)
    
    def forward(self,representation):
        s = representation.bmm(representation.transpose(1,2))
        for i in range(s.size(0)):
            for j in range(s.size(1)):
                s[i][j][j] = 0
        s = torch.softmax(s,-1)
        verify_rpt = s.bmm(representation)
        # p = torch.zeros(p_input_ids.size(0),p_input_ids.size(1)).to(device)
        p = F.softmax(self.w3_a(torch.cat((representation,verify_rpt,verify_rpt.mul(representation)),-1)),1)
        p = torch.log(p).squeeze(-1)
        return p

class BertForBaiduQA_Answer_Selection(BertPreTrainedModel):
# class BertForBaiduQA_Answer_Selection(AlbertPreTrainedModel):
    """ TBD """
    def __init__(self, config):
        super(BertForBaiduQA_Answer_Selection, self).__init__(config)
        self.temp_hidden = 200
        # self.bert =AlbertModel(config)
        # self.bert = BertModel(config)
        self.bert2 = Feature_Extraction(config)
        self.ptr = Ptr_net()
        self.content = Content_detect()
        self.verify = Verify_ans(config)
        self.init_weights()
        self.ptr.w2_a.weight.values = 10 * self.ptr.w2_a.weight
        self.ptr.w2_a.bias.values = 10 * self.ptr.w2_a.bias

    def part_one_loss(self,positions,probability,right_num):
        right_matrix = torch.zeros(probability.size()).to(probability.device)
        for i,(each_right_num) in enumerate(right_num):
            start = 0
            for each_doc_position in positions[i]:
                if(each_doc_position != 0):
                    start = each_doc_position
            right_matrix[start][i][each_right_num] = -1 * 1.0
        probability = probability.mul(right_matrix)
        start_loss = torch.max(torch.max(probability,0)[0],1)[0]
        final_loss = torch.div(start_loss.unsqueeze(0).mm(torch.ones(start_loss.size(0),1).to(probability.device)),start_loss.size(0))
        return final_loss

    def part_two_loss(self,start_positions,end_positions,probability,right_num):
        right_matrix =  torch.ones(probability.size()).to(probability.device)
        right_bias = torch.ones(probability.size()).to(probability.device)
        right_index = torch.zeros(probability.size(1),probability.size(2)).to(probability.device)
        for i,(each_right_num) in enumerate(right_num):
            for j,(each_start,each_end) in enumerate(zip(start_positions[i],end_positions[i])):
                if(each_start != 0 and each_end != 0):
                    right_index[i][j] = 1   
                    for temp_index in range(each_end-each_start+1):
                        right_matrix[each_start+temp_index][i][each_right_num] = -1 * 1.0    # 0-(-probability) 1-probabilty
                        right_bias[each_start+temp_index][i][each_right_num] = 0
                    break
                  
        final_loss =  probability.mul(right_matrix)
        final_loss = -1 * torch.log(right_bias - final_loss) 
        final_loss = torch.max(torch.sum(final_loss,0).mul(right_index),1)[0]
        final_loss = torch.div(final_loss,probability.size(0))
        return (final_loss[0]+final_loss[1])/2,right_index
    
    # def part_one_score(self,probability)
    
    def second_score(self,poss,start_positions,end_positions):
        scores = torch.zeros(end_positions.size())
        for i,(start,end) in enumerate(zip(start_positions,end_positions)):
            for j,(each_start,each_end) in enumerate(zip(start,end)):
                # a = torch.sum(poss[each_start:each_end+1,i,j],dim=0)
                scores[i][j] = torch.div(torch.sum(poss[each_start:each_end+1,i,j],dim=0),each_end+1-each_start)
                # return poss[each_start]
        return scores

    def forward(self, q_input_ids,  p_input_ids,q_attention_mask=None, q_token_type_ids=None, q_position_ids=None, q_head_mask=None,
                p_attention_mask=None, p_token_type_ids=None, p_position_ids=None, p_head_mask=None,right_num = None,
                start_positions=None, end_positions=None): 
        # device = p_input_ids.device
        """transpose batch and docs"""
        p_input_ids = p_input_ids.transpose(0,1)
        p_attention_mask = p_attention_mask.transpose(0,1)
        p_token_type_ids = p_token_type_ids.transpose(0,1)
        # batch_size = p_input_ids.size(1)
        q_features,q_embedding = self.bert2(q_input_ids,attention_mask = q_attention_mask,token_type_ids=q_token_type_ids,position_ids=q_position_ids, head_mask=q_head_mask)
        p0_features,p0_embedding= self.bert2(p_input_ids[0],attention_mask = p_attention_mask[0],token_type_ids=p_token_type_ids[0],position_ids=p_position_ids,head_mask=p_head_mask)
        p1_features,p1_embedding = self.bert2(p_input_ids[1],attention_mask = p_attention_mask[1],token_type_ids=p_token_type_ids[1],position_ids=p_position_ids,head_mask=p_head_mask)
        p2_features,p2_embedding = self.bert2(p_input_ids[2],attention_mask = p_attention_mask[2],token_type_ids=p_token_type_ids[2],position_ids=p_position_ids,head_mask=p_head_mask)
        # p_features = torch.cat((p0_features,p1_features,p2_features),1)
        # q_features_f = q_features.repeat(1,int(p_features.size(1)/q_features.size(1)),1)
        # final_f,p_alpha1,p_alpha2 = self.ptr(p_features,q_features_f)
        # p_embedding = torch.cat((p0_embedding,p1_embedding,p2_embedding),0)
        final_p0_features,p0_alpha1,p0_alpha2 = self.ptr(p0_features,q_features)
        final_p1_features,p1_alpha1,p1_alpha2 = self.ptr(p1_features,q_features)
        final_p2_features,p2_alpha1,p2_alpha2 = self.ptr(p2_features,q_features)
        alpha_1 = torch.cat((p0_alpha1.unsqueeze(-1).transpose(0,1),p1_alpha1.unsqueeze(-1).transpose(0,1),
                            p2_alpha1.unsqueeze(-1).transpose(0,1)),-1)
        alpha_2 = torch.cat((p0_alpha2.unsqueeze(-1).transpose(0,1),p1_alpha2.unsqueeze(-1).transpose(0,1),
                            p2_alpha2.unsqueeze(-1).transpose(0,1)),-1)

        p0_poss,p0_representation = self.content(final_p0_features,p0_embedding)
        p1_poss,p1_representation = self.content(final_p1_features,p1_embedding)
        p2_poss,p2_representation = self.content(final_p2_features,p2_embedding)
        poss = torch.cat((p0_poss,p1_poss,p2_poss),-1).transpose(0,1) # 400 2 3

        representation = torch.cat((p0_representation.unsqueeze(-1),p1_representation.unsqueeze(-1),p2_representation.unsqueeze(-1)),-1).transpose(1,2).transpose(0,1)
        p = self.verify(representation)
     
        if start_positions is not None and end_positions is not None:
            start_loss = self.part_one_loss(start_positions,alpha_1,right_num)
            end_loss = self.part_one_loss(end_positions,alpha_2,right_num)
            
            content_loss,right_index = self.part_two_loss(start_positions,end_positions,poss,right_num)

            part_three_loss = torch.max(p.transpose(0,1).mul(right_index),1)[0]
            verify_loss = (part_three_loss[0]+part_three_loss[1])/2

            return (start_loss +end_loss)/2 + 0.5 * content_loss + 0.5 * verify_loss 
        else:
            ans_start = torch.max(alpha_1,dim=0)
            ans_end = torch.max(alpha_2,dim=0)
            part_one_score = torch.exp(ans_start[0].float()).mul(torch.exp(ans_end[0].float()))
            part_two_score = self.second_score(poss,ans_start[1],ans_end[1])
            part_three_score = torch.exp(p).transpose(0,1)
            final_score = part_one_score + part_two_score + part_three_score
            indexs = torch.max(final_score,dim=1)[1]
            final_position = []
            for i in range(len(indexs)):
                final_position.append({'id':indexs[i],'start':ans_start[1][i][indexs[i]],'end':ans_end[1][i][indexs[i]]}) 
            return final_position
            