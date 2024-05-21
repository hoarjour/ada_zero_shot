import copy
from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import fvcore.nn.weight_init as weight_init


class SemanticsModel(nn.Module):

    def __init__(self, num_queries=100, d_model=512, nhead=8,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, num_classes=-1, text_feature_len=-1, image_feature_channel=2048,
                 channel_self_attention=False, window_size=196):
        super().__init__()
        self.channel_self_attention = channel_self_attention
        if self.channel_self_attention:
            d_model = window_size
            group_num = int(np.sqrt(window_size))
            nhead = group_num // 2
        else:
            group_num = 32

        self.return_intermediate = return_intermediate_dec
        self.image_proj = nn.Conv2d(image_feature_channel, d_model, kernel_size=1)
        self.img_norm = nn.GroupNorm(group_num, d_model)
        nn.init.xavier_uniform_(self.image_proj.weight, gain=1)
        nn.init.constant_(self.image_proj.bias, 0)

        self.query_embed = nn.Embedding(num_queries, d_model)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        self.semantics_proj = nn.Linear(d_model, 1)

        self.class_head = nn.Linear(num_queries, num_classes)
        self.semantics_align = nn.Linear(num_queries, text_feature_len)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        weight_init.c2_xavier_fill(self.semantics_proj)
        weight_init.c2_xavier_fill(self.class_head)
        weight_init.c2_xavier_fill(self.semantics_align)
        for name, p in self.named_parameters():
            if p.dim() > 1 and name.split('.')[0] == 'decoder':
                nn.init.xavier_uniform_(p)

    def forward(self, src):
        bs, c, h, w = src.shape
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)

        # 处理输入数据
        image_feature = self.image_proj(src)
        image_feature = self.img_norm(image_feature)
        memory = image_feature.flatten(2).permute(2, 0, 1)
        if self.channel_self_attention:
            memory = memory.permute(2, 1, 0)

        # mask 全为0
        mask = torch.zeros((bs, memory.shape[0]), device=memory.device).bool()

        # imagenet提取的特征，位置编码为空   # TODO 有待考证
        pos_embed = None

        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)

        if not self.return_intermediate:
            hs = hs[0]
        else:
            raise NotImplementedError("目前没有使用auxiliary_loss")

        hs = hs.permute(1, 0, 2)
        semantics_vector = torch.squeeze(self.semantics_proj(hs))

        # semantics_vector = torch.sigmoid(semantics_vector)  # TODO 这个地方可能是个大问题

        text_align_vector = self.semantics_align(semantics_vector)
        class_logits = self.class_head(semantics_vector)

        return semantics_vector, text_align_vector, class_logits


class TestModel(SemanticsModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_proj2 = nn.Linear(256 * 14 * 14, 100)

    def forward(self, src):
        bs, c, h, w = src.shape
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt = torch.zeros_like(query_embed)

        # 处理输入数据
        image_feature = self.image_proj(src)
        image_feature = self.img_norm(image_feature)
        memory = image_feature.view(bs, -1)
        x = self.image_proj2(memory)
        semantics_vector = F.relu(x)

        class_logits = self.class_head(x)
        text_align_vector = self.semantics_align(semantics_vector)
        return semantics_vector, text_align_vector, class_logits

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
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

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


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


def build_semantics_model(args):
    if args.dataset == 'AWA2':
        pass
    elif args.dataset == 'CUB':
        num_classes = 150
    elif args.dataset == 'SUN':
        pass
    else:
        raise ValueError(f"不支持的数据集名{args.dataset}")

    if args.class_feature_type == 'w2v':
        text_feature_len = 319
    elif args.class_feature_type == 'clip':
        text_feature_len = 512
    else:
        raise ValueError(f"不支持的class_feature_type {args.class_feature_type}")

    model = SemanticsModel(
        num_queries=args.num_queries,
        d_model=args.d_model,
        nhead=args.nhead,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation=args.activation,
        num_classes=num_classes,
        text_feature_len=text_feature_len,
        channel_self_attention=args.channel_self_attention,
        window_size=args.window_size
    )
    return model
