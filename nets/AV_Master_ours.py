import torch
# import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import copy
import math


from nets.bert_module import BertLMPredictionHead, BertLayer_Cross
from easydict import EasyDict as EDict

from nets.model import AVFC, AugEncoder

def cosine_similarity(a, b):

    a_norm = torch.linalg.norm(a, dim = -1, keepdim = True)
    a = a / a_norm

    b_norm = torch.linalg.norm(b, dim = -1, keepdim = True)
    b = b / b_norm

    return  torch.mm(a, b.t())

class TemporalSampling(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.class_embedding = nn.Parameter((width ** -0.5) * torch.randn(width))
        self.positional_embedding = nn.Parameter((width ** -0.5) * torch.randn(100, width))
        self.bert_config = EDict(
            num_attention_heads=8,
            hidden_size=width,
            attention_head_size=width,
            attention_probs_dropout_prob=0.1,
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.1,
            intermediate_size=width,
            vocab_size=42,
            num_layers=2
        )
        self.layer_ca = nn.ModuleList([BertLayer_Cross(self.bert_config) for _ in range(self.bert_config.num_layers)])
        self.head = BertLMPredictionHead(self.bert_config)

    def forward(self, x, query=None):
        #x = F.adaptive_avg_pool2d(x, (1, 1)).squeeze().unsqueeze(0)
        for i in range(self.bert_config.num_layers):
            x, _ = self.layer_ca[i](x, query)
        
        logits = self.head(x).squeeze()
        return logits

class SpatialActivation(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.vocab_size = 42
        self.class_embedding = nn.Parameter((width ** -0.5) * torch.randn(width))
        self.positional_embedding = nn.Parameter((width ** -0.5) * torch.randn(100, width))
        self.bert_config = EDict(
            num_attention_heads=8,
            hidden_size=width,
            attention_head_size=width,
            attention_probs_dropout_prob=0.1,
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.1,
            intermediate_size=width,
            vocab_size=42,
            num_layers=2
        )
        self.layer_ca = nn.ModuleList([BertLayer_Cross(self.bert_config) for _ in range(self.bert_config.num_layers)])
        self.head = BertLMPredictionHead(self.bert_config)

    def forward(self, input, init_q=None):
        #input = input.permute(0, 2, 3, 1)
        #x = input.reshape(input.size(0), -1, 256)
        #query = torch.zeros(x.size(0), 1, x.size(-1)).to(x.device) if init_q is None else init_q.repeat(x.size(0), 1, 1)
        x = input
        query = init_q
        for i in range(self.bert_config.num_layers):
            query, att_map = self.layer_ca[i](query, x)
        att_map = att_map.sum(1).squeeze(1).sigmoid()
        att_map = (att_map - att_map.min(dim=1, keepdim=True)[0]) / (att_map.max(dim=1, keepdim=True)[0] - att_map.min(dim=1, keepdim=True)[0])
        
        #logits = self.head(query).mean(0)
        logits = self.head(query).squeeze()
        #return logits, att_map
        return logits


class QstLstmEncoder(nn.Module):

    def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size):

        super(QstLstmEncoder, self).__init__()
        self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(2*num_layers*hidden_size, embed_size)     # 2 for hidden and cell states

    def forward(self, question):

        qst_vec = self.word2vec(question)                             # [batch_size, max_qst_length=30, word_embed_size=300]
        qst_vec = self.tanh(qst_vec)
        qst_vec = qst_vec.transpose(0, 1)                             # [max_qst_length=30, batch_size, word_embed_size=300]
        self.lstm.flatten_parameters()
        _, (hidden, cell) = self.lstm(qst_vec)                        # [num_layers=2, batch_size, hidden_size=512]
        qst_feature = torch.cat((hidden, cell), 2)                    # [num_layers=2, batch_size, 2*hidden_size=1024]
        qst_feature = qst_feature.transpose(0, 1)                     # [batch_size, num_layers=2, 2*hidden_size=1024]
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)                            # [batch_size, embed_size]

        return qst_feature


class AVClipAttn(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(AVClipAttn, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cm_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src_q, src_v, src_mask=None, src_key_padding_mask=None):

        src_q = src_q.permute(1, 0, 2)
        src_v = src_v.permute(1, 0, 2)
        src1 = self.cm_attn(src_q, src_v, src_v, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
        src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]

        src_q = src_q + self.dropout11(src1) + self.dropout12(src2)
        src_q = self.norm1(src_q)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
        src_q = src_q + self.dropout2(src2)
        src_q = self.norm2(src_q)

        return src_q.permute(1, 0, 2)






class AVHanLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(AVHanLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cm_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src_q, src_v, src_mask=None, src_key_padding_mask=None):

        src_q = src_q.permute(1, 0, 2)
        src_v = src_v.permute(1, 0, 2)
        src1 = self.cm_attn(src_q, src_v, src_v, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src_q = src_q + self.dropout11(src1) + self.dropout12(src2)
        src_q = self.norm1(src_q)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
        src_q = src_q + self.dropout2(src2)
        src_q = self.norm2(src_q)
        return src_q.permute(1, 0, 2)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class GlobalLocalPrecption(nn.Module):

    def __init__(self, args, encoder_layer, num_layers, norm=None):
        super(GlobalLocalPrecption, self).__init__()

        self.args = args

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.norm = norm

    def forward(self, src_a, src_v, mask=None, src_key_padding_mask=None):
        
        #audio_output = src_a
        #visual_output = src_v

        for i in range(self.num_layers):
            src_a = self.layers[i](src_a, src_v, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            #src_v = self.layers[i](src_v, src_a, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        #if self.norm:
        #    src_a = self.norm1(src_a)
        #    src_v = self.norm2(src_v)

        return src_a





class GlobalHanLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(GlobalHanLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cm_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src_q, src_v, src_mask=None, src_key_padding_mask=None):

        src_q = src_q.permute(1, 0, 2)
        src_v = src_v.permute(1, 0, 2)
        src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src_q = src_q + self.dropout12(src2)
        src_q = self.norm1(src_q)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
        src_q = src_q + self.dropout2(src2)
        src_q = self.norm2(src_q)
        return src_q.permute(1, 0, 2)

class GlobalSelfAttn(nn.Module):

    def __init__(self, args, encoder_layer, num_layers, norm=None):
        super(GlobalSelfAttn, self).__init__()

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm2 = nn.LayerNorm(512)
        self.norm = norm

    def forward(self, src_v, mask=None, src_key_padding_mask=None):
        
        visual_output = src_v

        for i in range(self.num_layers):
            visual_output = self.layers[i](src_v, src_v, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm:
            visual_output = self.norm2(visual_output)

        return visual_output


class AV_Master(nn.Module):

    def __init__(self, args, hidden_size=768):
        super(AV_Master, self).__init__()

        self.args = args
        self.num_layers = args.num_layers
    
        self.fc_a =  nn.Linear(128, hidden_size)
        self.fc_v = nn.Linear(768, hidden_size)
        self.fc_p = nn.Linear(768, hidden_size)
        self.fc_word = nn.Linear(768, hidden_size)

        self.relu = nn.ReLU()

        if self.args.question_encoder == "CLIP":
            self.fc_q = nn.Linear(768, hidden_size)
        else:
            self.fc_q = QstLstmEncoder(93, 512, 512, 1, 512)

        self.fc_spat_q = nn.Linear(768, hidden_size)

            
        # modules
        #self.TempSegsSelect_Module = TemporalSegmentSelection(args)
        #self.SpatRegsSelect_Module = SpatioRegionSelection(args)
        #self.AudioGuidedVisualAttn = AudioGuidedVisualAttn(args)

        #self.GlobalLocal_Module = GlobalLocalPrecption(args, 
        #                                               AVHanLayer(d_model=768, nhead=1, dim_feedforward=768), 
        #                                               num_layers=self.num_layers)
        self.GlobalSelf_Module = GlobalSelfAttn(args, 
                                                GlobalHanLayer(d_model=768, nhead=1, dim_feedforward=768), 
                                                num_layers=self.num_layers)
        self.AP_Module = GlobalLocalPrecption(args, 
                                                       AVHanLayer(d_model=768, nhead=1, dim_feedforward=768), 
                                                       num_layers=self.num_layers)
        self.VP_Module = GlobalLocalPrecption(args, 
                                                       AVHanLayer(d_model=768, nhead=1, dim_feedforward=768), 
                                                       num_layers=self.num_layers)
        
        self.AugInformation = AugEncoder(AV_Encoder= AVFC(d_model=768),
                                        lens= 8,
                                        feature_dim = 768,
                                        object_dim = 768,
                                        hidden_dim = 768)
        self.bisa_v = nn.Embedding(8, 768)
        self.bias_a = nn.Embedding(8, 768)

        # fusion with audio and visual feat
        self.audio_fusion = nn.Linear(768, 768)
        self.visual_fusion = nn.Linear(768, 768)

        self.SM_fusion = nn.Linear(1024, 768)

        self.tanh_av_fusion = nn.Tanh()
        self.fc_av_fusion = nn.Linear(1024, 768)
        self.tanh_avq_fusion = nn.Tanh()

        self.Uo_v = nn.Linear(768, 768)
        self.bo_v = nn.Parameter(torch.ones(768), requires_grad=True)
        self.wo_v = nn.Linear(768, 1)

        self.Uo_a = nn.Linear(768, 768)
        self.bo_a = nn.Parameter(torch.ones(768), requires_grad=True)
        self.wo_a = nn.Linear(768, 1)

        self.linear_visual_layer = nn.Linear(2304, 768)

        # answer prediction
        #self.fc_answer_pred = nn.Linear(512, 42)

        #self.fc_answer_pred = TemporalSampling(512)
        #self.fc_answer_pred2 = TemporalSampling(512)
        #self.fc_answer_pred3 = TemporalSampling(512)

        self.fc_answer_pred = SpatialActivation(768)
        self.fc_answer_pred2 = SpatialActivation(768)
        self.fc_answer_pred3 = SpatialActivation(768)



    def Fusion(self, visual, object_v, object_a):
        #if fusion_object:
        U_objs = self.Uo_v(object_v)
        attn_feat = visual.unsqueeze(2) + U_objs.unsqueeze(1) + self.bo_v  # (bsz, sample_numb, max_objects, hidden_dim)
        attn_weights = self.wo_v(torch.tanh(attn_feat))  # (bsz, sample_numb, max_objects, 1)
        attn_weights = attn_weights.softmax(dim=-2)  # (bsz, sample_numb, max_objects, 1)
        attn_objects = attn_weights * attn_feat
        attn_objects = attn_objects.sum(dim=-2)  # (bsz, sample_numb, hidden_dim)

        U_objs_a = self.Uo_a(object_a)
        attn_feat_a = visual.unsqueeze(2) + U_objs_a.unsqueeze(1) + self.bo_a  # (bsz, sample_numb, max_objects, hidden_dim)
        attn_weights_a = self.wo_a(torch.tanh(attn_feat_a))  # (bsz, sample_numb, max_objects, 1)
        attn_weights_a = attn_weights_a.softmax(dim=-2)  # (bsz, sample_numb, max_objects, 1)
        attn_objects_a = attn_weights_a * attn_feat_a
        attn_objects_a = attn_objects_a.sum(dim=-2)  # (bsz, sample_numb, hidden_dim)

        features = torch.cat([visual, attn_objects, attn_objects_a], dim=-1)
        output = self.linear_visual_layer(features)
        context = torch.max(output, dim=1)[0]  # (bsz, hidden_dim)  
        return context

    def forward(self, audio, visual, patch, question, qst_word, neg_visual_feat=None, neg_audios_feat=None):

        ### 1. features input 
        # audio: [B, T, C]
        # visual: [B, T, C]
        # question: [B, C]
        # patch: [B, T, N, C], N: patch numbers

        if len(audio.size()) > 3:
            audio = audio.squeeze()
            if neg_audios_feat!=None:
                neg_audios_feat = neg_audios_feat.squeeze()

        audio_feat = self.fc_a(audio)                   # [B, T, C]
        visual_feat = self.fc_v(visual)                 # [B, T, C]
        
        if self.args.use_word:
            word_feat = self.fc_word(qst_word).squeeze(-3)  # [B, 77, C]
        qst_feat = self.fc_q(question).squeeze(-2)      # [B, C]

        contrastive_loss = None

        TDPP_visual= self.AugInformation(
            visual = visual_feat,
            pos = self.bisa_v.weight
        )
        TDPP_audio= self.AugInformation(
            visual = audio_feat,
            pos = self.bias_a.weight
        )

        fusion_feat = torch.cat((audio_feat, visual_feat, word_feat), dim=1)
        av_fusion_feat = self.GlobalSelf_Module(fusion_feat)
        audio_pre = self.AP_Module(audio_feat, word_feat)
        visual_pre = self.VP_Module(visual_feat, word_feat)
        av_fusion_feat = self.Fusion(av_fusion_feat, TDPP_visual, TDPP_audio)

#        if neg_visual_feat != None and neg_audios_feat !=None:
#            neg_audio_feat = self.fc_a(neg_audios_feat)         
#            neg_visual_feat = self.fc_v(neg_visual_feat)  
#            neg_fusion_feat_a = torch.cat((neg_audio_feat, word_feat), dim=1)
#            neg_fusion_feat_v = torch.cat((neg_visual_feat, word_feat), dim=1)  
#            neg_fusion_feat_a = self.GlobalSelf_Module(neg_fusion_feat_a)
#            neg_fusion_feat_v = self.GlobalSelf_Module(neg_fusion_feat_v) 
#            neg_fusion_feat_a = self.Uo(neg_fusion_feat_a)
#            neg_fusion_feat_v = self.Uo(neg_fusion_feat_v)
#
#            neg_attn_feat_a = av_fusion_feat.unsqueeze(1).unsqueeze(2) + neg_fusion_feat_a.unsqueeze(1) + self.bo  
#            neg_sem_align_logits_a = self.wo(torch.tanh(neg_attn_feat_a)).squeeze(-1) 
#            neg_attn_feat_v = av_fusion_feat.unsqueeze(1).unsqueeze(2) + neg_fusion_feat_v.unsqueeze(1) + self.bo  
#            neg_sem_align_logits_v = self.wo(torch.tanh(neg_attn_feat_v)).squeeze(-1)  
#
#            pos_attn_feat_a = av_fusion_feat.unsqueeze(1).unsqueeze(2) + av_fusion_feat2.unsqueeze(1) + self.bo  
#            pos_sem_align_logits_a = self.wo(torch.tanh(pos_attn_feat_a)).squeeze(-1)  
#            pos_attn_feat_v = av_fusion_feat.unsqueeze(1).unsqueeze(2) + av_fusion_feat3.unsqueeze(1) + self.bo  
#            pos_sem_align_logits_v = self.wo(torch.tanh(pos_attn_feat_v)).squeeze(-1)  
#
#            neg_sem_align_logits = neg_sem_align_logits_a + neg_sem_align_logits_v
#            pos_sem_align_logits = pos_sem_align_logits_a + pos_sem_align_logits_v
#
#            pos_align_logit = pos_sem_align_logits.sum(dim=2) 
#            neg_align_logit = neg_sem_align_logits.sum(dim=2)
#            align_logits = torch.stack([ pos_align_logit, neg_align_logit ], dim=2)
#            align_logits = align_logits.view(-1, 2)
#
#            contrastive_loss = F.binary_cross_entropy_with_logits(
#                align_logits.mean(dim=0), torch.cuda.FloatTensor([ 1, 0 ]))
        if neg_visual_feat != None and neg_audios_feat !=None:
            tau = 0.1

            neg_audio_feat = self.fc_a(neg_audios_feat)         
            neg_visual_feat = self.fc_v(neg_visual_feat)  

            neg_fusion_feat_a = self.AP_Module(neg_audio_feat, word_feat)
            #neg_fusion_feat_a = torch.cat((n_g_a, n_g_a_que), dim=1)

            neg_fusion_feat_v = self.VP_Module(neg_visual_feat, word_feat)
            #neg_fusion_feat_v = torch.cat((n_g_v, n_g_v_que), dim=1)

            neg_fusion_feat_a = neg_fusion_feat_a.mean(dim=-2)
            neg_fusion_feat_v = neg_fusion_feat_v.mean(dim=-2)

            sim_positive = cosine_similarity(av_fusion_feat, audio_pre.mean(dim=-2)) / tau + cosine_similarity(av_fusion_feat, visual_pre.mean(dim=-2)) / tau
            p_i_1 = torch.exp(sim_positive)

            neg_sim_negative = cosine_similarity(av_fusion_feat, neg_fusion_feat_a) / tau + cosine_similarity(av_fusion_feat, neg_fusion_feat_v) / tau
            p_neg_sum = torch.exp(neg_sim_negative)

            contrastive_loss = -torch.log(p_i_1 / (p_neg_sum + p_i_1))
            contrastive_loss = contrastive_loss.mean()

        av_fusion_feat = av_fusion_feat.squeeze().unsqueeze(1)
        answer_pred = self.fc_answer_pred(av_fusion_feat,qst_feat.squeeze().unsqueeze(1))  # [batch_size, ans_vocab_size=42]
        answer_pred2 = self.fc_answer_pred2(audio_pre,qst_feat.squeeze().unsqueeze(1))
        answer_pred3 = self.fc_answer_pred3(visual_pre,qst_feat.squeeze().unsqueeze(1))

        return answer_pred,answer_pred2,answer_pred3, contrastive_loss
