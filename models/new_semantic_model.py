import copy
from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import fvcore.nn.weight_init as weight_init

from models.image_backbone import build_image_backbone
from models.transformer import build_transformer
from utils.misc import NestedTensor, nested_tensor_from_tensor_list

dataset_attr_len = {
    "CUB": {
        "num_classes": 150,
        "text_feature_len": 309
    },
    "SUN": {
        "num_classes": 645,
        "text_feature_len": 300
    },
    "AWA2": {
        "num_classes": 40,
        "text_feature_len": 100
    }

}

class NewSemanticsModel(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss, text_feature_len,
                 channel_self_attention=False, window_size=196):
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.aux_loss = aux_loss
        self.channel_self_attention = channel_self_attention
        if self.channel_self_attention:
            d_model = window_size
            group_num = int(np.sqrt(window_size))
        else:
            d_model = transformer.d_model
            group_num = 32

        image_feature_channel = backbone.num_channels

        self.image_proj = nn.Conv2d(image_feature_channel, d_model, kernel_size=1)
        self.img_norm = nn.GroupNorm(group_num, d_model)
        nn.init.xavier_uniform_(self.image_proj.weight, gain=1)
        nn.init.constant_(self.image_proj.bias, 0)

        self.query_embed = nn.Embedding(num_queries, d_model)

        self.semantics_proj = nn.Linear(d_model, 1)

        self.class_head = nn.Linear(num_queries, num_classes)
        # self.box_head = MLP(d_model, d_model, 4, 3)
        self.box_head = nn.Linear(d_model, 4)
        self.semantics_align = nn.Linear(num_queries, text_feature_len)

        self._reset_parameters()

        self.d_model = d_model

    def _reset_parameters(self):
        weight_init.c2_xavier_fill(self.semantics_proj)
        weight_init.c2_xavier_fill(self.class_head)
        weight_init.c2_xavier_fill(self.semantics_align)
        for name, p in self.named_parameters():
            if p.dim() > 1 and name.split('.')[0] == 'transformer':
                nn.init.xavier_uniform_(p)

    def forward(self, samples):
        if isinstance(samples, (list, torch.Tensor)):
            raise Exception
            # samples = nested_tensor_from_tensor_list(samples)
        # 图像正向过程
        features, pos = self.backbone(samples)
        image_src, image_mask = features[-1].decompose()

        # 图像维度映射
        image_src = self.image_proj(image_src)  # (batch_size, channel, h, w)
        image_src = self.img_norm(image_src)
        # bs, c, h, w = image_src.shape
        # image_src = image_src.flatten(2).permute(2, 0, 1)  # (channel, batch_size, hidden_dim)
        # pos_embed = pos[-1].flatten(2).permute(2, 0, 1)  # (channel, batch_size, hidden_dim)
        # image_mask = image_mask.flatten(1)  # (batch_size, channel)

        # object query初始化
        query_embed = self.query_embed.weight
        hs = self.transformer(image_src, image_mask, query_embed, pos[-1])[0] # 2, batch_size, 100, 256

        if not self.aux_loss:
            hs = hs[-1]
        else:
            raise NotImplementedError('现在不支持aux_loss')

        boxes_output = self.box_head(hs).sigmoid()

        semantics_vector = torch.squeeze(self.semantics_proj(hs))

        # semantics_vector = torch.sigmoid(semantics_vector)  # TODO 这个地方可能是个大问题

        text_align_vector = self.semantics_align(semantics_vector)
        class_logits = self.class_head(semantics_vector)

        return semantics_vector, text_align_vector, class_logits, boxes_output


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_new_semantics_model(args):
    backbone = build_image_backbone(args)

    transformer = build_transformer(args)

    num_classes = dataset_attr_len[args.dataset]['num_classes']
    if args.class_feature_type == 'clip':
        text_feature_len = 512
    elif args.class_feature_type == 'w2v':
        text_feature_len = dataset_attr_len[args.dataset]['text_feature_len']
    else:
        raise ValueError(f"不支持的class_feature_type {args.class_feature_type}")

    model = NewSemanticsModel(backbone=backbone,
                              transformer=transformer,
                              num_classes=num_classes,
                              num_queries=args.num_queries,
                              aux_loss=args.aux_loss,
                              text_feature_len=text_feature_len,
                              channel_self_attention=args.channel_self_attention,
                              window_size=args.window_size
                              )
    return model

