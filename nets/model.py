import torch
import torch.nn.functional as F
from typing import Optional, List
from torch import nn, Tensor
import copy


class AVFC(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=1,
                 num_decoder_layers=1, dim_feedforward=512, dropout=0.1,
                 activation="gelu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TrackerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TrackerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TrackerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TrackerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, visual, bisa, lens, mask=None):
        """ AVFC """
        """
        src:visual/audio
        tgt:query_embed
        """
        bs = visual.shape[0]  # key_visual=[B,L,D]
        v_len = visual.shape[1]
        templates = visual[:, 0, :].unsqueeze(1).repeat(1, lens, 1)
        for j in range(v_len):
            visual_frame = visual[:, j, :].unsqueeze(1).repeat(1, lens, 1)  # (max_objects, bsz, hidden_dim)

            templates = self.encoder(templates, visual_frame, mask=mask)

            templates = templates.permute(1, 0, 2)  # [N,B,D]
            query_bisa = bisa.unsqueeze(1).repeat(1, bs, 1)  # [N,B,D]
            if mask is not None:
                mask = mask.flatten(1)

            templates = self.decoder(templates, visual_frame.permute(1, 0, 2), query_pos=query_bisa, tgt_mask=mask)
            templates = templates.permute(1, 0, 2)
        return templates

class TrackerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src1, src2, mask: Optional[Tensor] = None):
        output = src1

        for layer in self.layers:
            output = layer(output, src2)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TrackerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TrackerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(d_model, dim_feedforward)
        self.b = nn.Parameter(torch.ones(dim_feedforward), requires_grad=True)
        self.w = nn.Linear(dim_feedforward, 1)
        self.linear3 = nn.Linear(dim_feedforward, d_model)

    def forward(self, src1, src2):
        tgt = self.linear1(src1)  # (bsz, sample_numb, hidden_dim)
        query = self.linear2(src2)  # (bsz, max_objects, hidden_dim)

        attn_tgt = tgt.unsqueeze(2) + query.unsqueeze(1) + self.b  # (bsz, max_objects, sample_num, hidden_dim)
        attn_weights = self.w(torch.tanh(attn_tgt))  # (bsz, max_objects, sample_num, 1)
        attn_weights = attn_weights.softmax(dim=-2)  # (bsz, max_objects, sample_num, 1)
        tgt1 = attn_weights * attn_tgt
        tgt1 = tgt1.sum(dim=-2)  # (bsz, max_objects, hidden_dim)
        tgt = self.linear3(tgt1)
        return tgt


class TrackerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        if query_pos is None:
            tgt_self = tgt
        else:
            tgt_self = tgt+query_pos
        tgt1 = self.self_attn(tgt_self, tgt_self, value=tgt, attn_mask=None, key_padding_mask=None)[0]
        tgt = tgt + self.dropout1(tgt1)
        tgt = self.norm1(tgt)

        if query_pos is None:
            tgt_cross = tgt
        else:
            tgt_cross = tgt+query_pos
        tgt2 = self.cross_attn(query=tgt_cross,
                                   key=memory,
                                   value=memory, attn_mask=None,
                                   key_padding_mask=None)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class Norm(nn.Module):
    def __init__(self, d_model=768):
        super().__init__()

        self._reset_parameters()
        #
        self.d_model = d_model

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt):

        tgt_norm = tgt / tgt.norm(dim=-1, keepdim=True)  # (64, 8, 512)

        cosine_sim_matrix = torch.matmul(tgt_norm, tgt_norm.transpose(1, 2))  # (64, 8, 10)

        inverse_similarity_matrix = cosine_sim_matrix
        inverse_similarity_matrix = torch.tanh(inverse_similarity_matrix)

        weighted_tgt = torch.bmm(inverse_similarity_matrix, tgt)
        output = weighted_tgt

        return output
    
class AugEncoder(nn.Module):
    def __init__(self, AV_Encoder, lens, feature_dim, object_dim,
                                           hidden_dim):
        super(AugEncoder, self).__init__()
        self.video_embeddings = nn.Linear(feature_dim, hidden_dim)
        if object_dim is not None:
            self.object_embeddings = nn.Linear(object_dim, hidden_dim)
        self.lens = lens
        self.focus_capture = AV_Encoder
        self.norm = Norm()

    def forward(self, visual, pos = None):
        vhidden_states = self.video_embeddings(visual)

        if self.focus_capture is not None:
            focus_hidden_states= self.focus_capture(vhidden_states, pos, self.lens, mask=None)
        else:
            focus_hidden_states = vhidden_states

        focus_hidden_states = self.norm(focus_hidden_states)

        return focus_hidden_states

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


