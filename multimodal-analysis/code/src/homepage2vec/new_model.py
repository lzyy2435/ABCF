# -*- coding: utf-8 -*-
# @Time    : 2023-07-15 16:30
# @Author  : Wei Liu
# @ID      : 2020212172
# @File    : new_model.py


import logging
import random

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from cnsn import CrossNorm, SelfNorm, CNSN, Get_CNSN


# TFN融合策略
# TFN来自17年EMNLP会议论文《Tensor Fusion Network for Multimodal Sentiment Analysis》，
# 其主要考虑了inter-modality和intar-modality两个方面。
# 也就是要求既能考虑各模态之间的特征融合，也要有效地利用各特定模态的特征。

class TFN(nn.Module):
    def __init__(self):
        super(TFN, self).__init__()
        # self.layerAB = torch.nn.Linear(795+798, 224)
        # self.layerBC = torch.nn.Linear(798+768, 224)
        # self.layerCA = torch.nn.Linear(768+795, 224)


    def forward(self, A,B,C):
        n = A.shape[0]

        A = torch.cat([A, torch.ones(n, 3).to('cuda:0')], dim=1)
        C = torch.cat([C, torch.ones(n, 30).to('cuda:0')], dim=1)
        # AB = torch.cat([A, B], dim=1)
        # BC = torch.cat([B, C], dim=1)
        # CA = torch.cat([C, A], dim=1)

        # A = self.layerAB(AB)
        # B = self.layerBC(BC)
        # C = self.layerCA(CA)

        # 计算笛卡尔积
        fusion_AB = torch.einsum('nxt, nty->nxy', A.unsqueeze(2), B.unsqueeze(1))  # [n, A, B]
        fusion_BC = torch.einsum('nxt, nty->nxy', B.unsqueeze(2), C.unsqueeze(1))  # [n, A, B]
        fusion_CA = torch.einsum('nxt, nty->nxy', C.unsqueeze(2), A.unsqueeze(1))  # [n, A, B]

        fusion_ABC = torch.stack([fusion_AB,fusion_BC,fusion_CA])

        # fusion_ABC = torch.stack([A,B,C])
        # fusion_ABC = fusion_ABC.flatten(start_dim=1)  # [n, AxBxC]
        # A, B, C分别代表原来的特征维度nA,nB,nC加上1
        fusion_ABC = fusion_ABC.reshape((n, 3, 798, 798))
        # print("              out:", fusion_ABC.shape)
        return fusion_ABC

class TFN2(nn.Module):
    def __init__(self):
        super(TFN2, self).__init__()
        self.layerAB = torch.nn.Linear(795+798, 224)
        self.layerBC = torch.nn.Linear(798+768, 224)
        self.layerCA = torch.nn.Linear(768+795, 224)


    def forward(self, A,B,C):
        n = A.shape[0]

        AB = torch.cat([A, B], dim=1)
        BC = torch.cat([B, C], dim=1)
        CA = torch.cat([C, A], dim=1)

        AB = F.relu(AB)
        BC = F.relu(BC)
        CA = F.relu(CA)

        A = self.layerAB(AB)
        B = self.layerBC(BC)
        C = self.layerCA(CA)

        # 计算笛卡尔积
        fusion_AB = torch.einsum('nxt, nty->nxy', A.unsqueeze(2), B.unsqueeze(1))  # [n, A, B]
        fusion_BC = torch.einsum('nxt, nty->nxy', B.unsqueeze(2), C.unsqueeze(1))  # [n, A, B]
        fusion_CA = torch.einsum('nxt, nty->nxy', C.unsqueeze(2), A.unsqueeze(1))  # [n, A, B]

        fusion_ABC = torch.stack([fusion_AB,fusion_BC,fusion_CA])

        # fusion_ABC = torch.stack([A,B,C])
        # fusion_ABC = fusion_ABC.flatten(start_dim=1)  # [n, AxBxC]
        # A, B, C分别代表原来的特征维度nA,nB,nC加上1
        fusion_ABC = fusion_ABC.reshape((n, 3, 224, 224))
        # print("              out:", fusion_ABC.shape)
        return fusion_ABC

# 减小维度！！！！！！
# [768+27+1 ,768+30+1, 768+1]->[224, 224, 10]
# URL, html, text

class DemensionReduce(nn.Module):
    def __init__(self, input_dim=3, output_dim=3):
        # [768+27+1 ,768+30+1, 768+1]->[224, 224, 10]
        # URL, html, text
        super(DemensionReduce, self).__init__()
        # self.conv1_1 = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=1, stride=1)
        self.conv1_2 = nn.Conv2d(in_channels=output_dim, out_channels=output_dim, kernel_size=7, stride=3, padding=0)

    def forward(self, x):
        batch = x.shape[0]
        # x = torch.rand((batch, C, H, W))
        # x = self.conv1_1(x)
        x = self.conv1_2(x)
        # print(x.shape)
        return x


# 特征扩增
# [768+27+1 ,768+30+1, 768+1]->[224, 224, 10]
# URL, html, text

class FA(nn.Module):
    def __init__(self,try_type = 1, input_dim=796, output_dim=3):
        # [768+27+1 ,768+30+1, 768+1]->[224, 224, 10]
        # URL, html, text
        super(FA, self).__init__()
        if try_type == 1:
            self.TFN = TFN()
        elif try_type == 2:
            self.TFN = TFN2()
        # self.DemensionReduce = DemensionReduce(input_dim, output_dim)

    def forward(self, A, B, C):
        # batch = x.shape[0]
        # x = torch.rand((batch, C, H, W))
        x = self.TFN(A, B, C)
        # x = self.DemensionReduce(x)
        return x

class FA2(nn.Module):
    def __init__(self,try_type = 1, input_dim=796, output_dim=3):
        # [768+27+1 ,768+30+1, 768+1]->[224, 224, 10]
        # URL, html, text
        super(FA2, self).__init__()
        self.DemensionReduce = DemensionReduce(input_dim, output_dim)

    def forward(self, A, B, C):
        x = self.DemensionReduce(x)
        return x

'''-------------SE模块-----------------------------'''


# 全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid
class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=1):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        # 读取批数据图片数量及通道数
        b, c, h, w = x.size()
        # Fsq操作：经池化后输出b*c的矩阵
        y = self.gap(x).view(b, c)
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale操作：将得到的权重乘以原来的特征图x
        return x * y.expand_as(x)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


# SelfAttention
import math


class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob=0.5):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(hidden_dropout_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


# import torch.nn.Module
import torch


# import torch.nn.init
# def init_conv(conv, glu=True):
#     init.xavier_uniform_(conv.weight)
#     if conv.bias is not None:
#         conv.bias.data.zero_()

class SelfAttention_2D(nn.Module):
    r"""
        Self attention Layer.
        Source paper: https://arxiv.org/abs/1805.08318
    """

    def __init__(self, in_dim, activation=F.relu):
        super(SelfAttention_2D, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.f = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.g = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.h = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        # init_conv(self.f)
        # init_conv(self.g)
        # init_conv(self.h)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention feature maps
                
        """
        m_batchsize, C, width, height = x.size()

        f = self.f(x).view(m_batchsize, -1, width * height)  # B * (C//8) * (W * H)
        g = self.g(x).view(m_batchsize, -1, width * height)  # B * (C//8) * (W * H)
        h = self.h(x).view(m_batchsize, -1, width * height)  # B * C * (W * H)

        attention = torch.bmm(f.permute(0, 2, 1), g)  # B * (W * H) * (W * H)
        attention = self.softmax(attention)

        self_attetion = torch.bmm(h, attention)  # B * C * (W * H)
        self_attetion = self_attetion.view(m_batchsize, C, width, height)  # B * C * W * H

        out = self.gamma * self_attetion + x
        return out


class FEA(nn.Module):
    def __init__(self):
        super(FEA, self).__init__()
        self.se = SE_Block(inchannel=3)
        # self.sa = SelfAttention_2D(num_attention_heads=1,input_size=3, hidden_size=3)
        # self.sa = SelfAttention_2D(in_dim=3)
        # self.cnsn = CNSN()
        self.sn1 = Get_CNSN("sn", in_channels=3, crop="neither", beta=1)
        self.sn2 = Get_CNSN("sn", in_channels=3, crop="neither", beta=1)
        self.sn3 = Get_CNSN("sn", in_channels=3, crop="neither", beta=1)

    def forward(self, x):
        x = self.sn1(x)
        res = x
        x = self.se(x) + res
        x = self.sn2(x)
        # res = x
        # x = self.sa(x) + res
        x = self.sn3(x)
        return x
