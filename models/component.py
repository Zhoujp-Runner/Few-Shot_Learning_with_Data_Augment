# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 19:04 2023/3/27
"""
一些用于其他网络中的组件
"""

import torch
from torch import nn
import math


class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_condition):
        """
        :param dim_in: 输入x的维度
        :param dim_out: 输出的维度
        :param dim_condition: 输入条件的维度
        """
        super(ConcatSquashLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_gate = nn.Linear(dim_condition, dim_out)
        self._hyper_bias = nn.Linear(dim_condition, dim_out, bias=False)

    def forward(self, x, condition):
        gate = torch.sigmoid(self._hyper_gate(condition))
        bias = self._hyper_bias(condition)
        output = self._layer(x) * gate + bias
        return output


class SelfAttention(nn.Module):
    def __init__(self, dim_in, hidden_dim, att_drop_prob=0.5, out_drop_prob=0.5):
        """
        单头自注意力网络
        :param dim_in: 输入维度
        :param hidden_dim: 中间隐藏层维度
        :param att_drop_prob: 注意力dropout率
        :param out_drop_prob: 输出dropout率
        """
        super(SelfAttention, self).__init__()
        # Q, K, V三个矩阵的权重参数
        self.query_weight = nn.Linear(dim_in, hidden_dim)
        self.key_weight = nn.Linear(dim_in, hidden_dim)
        self.value_weight = nn.Linear(dim_in, hidden_dim)

        self.softmax = nn.Softmax(dim=-1)

        # TODO 下面这些存在的必要性需要通过消融研究一下
        self.att_drop = nn.Dropout(att_drop_prob)
        self.dense = nn.Linear(hidden_dim, dim_in)  # 使维度变回原来的维度，方便之后做残差连接
        # self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-12)
        self.out_drop = nn.Dropout(out_drop_prob)

        # 记录维度，用于后续的归一化
        self.hidden_dim = hidden_dim

    def forward(self, x):
        """
        x: [batch_size, attribute_num, attribute_dim]
        """
        # 得到 Q, K, V 三个矩阵
        query_matrix = self.query_weight(x)  # [batch_size, attribute_num, hidden_dim]
        key_matrix = self.key_weight(x)
        value_matrix = self.value_weight(x)

        attention_score = torch.matmul(query_matrix, key_matrix.transpose(-1, -2))
        attention_score = attention_score / math.sqrt(self.hidden_dim)
        attention_prob = self.softmax(attention_score)

        att_prob_with_drop = self.att_drop(attention_prob)

        context = torch.matmul(att_prob_with_drop, value_matrix)

        result = self.dense(context)
        result = self.out_drop(result)
        # result = self.layer_norm(result)

        return result


class FeedForward(nn.Module):
    def __init__(self, dim_in, hidden_dim):
        """
        全连接神经网络
        :param dim_in: 输入维度
        :param hidden_dim: 中间隐藏层维度
        """
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, dim_in)
        # self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-12)

    def forward(self, x):
        ffn = self.linear2(self.relu(self.linear1(x)))
        return ffn


class TransformerLayer(nn.Module):
    def __init__(self, dim_in, att_hidden_dim, ffn_hidden_dim):
        super(TransformerLayer, self).__init__()
        self.self_attention = SelfAttention(dim_in, att_hidden_dim)
        self.layer_norm = nn.LayerNorm(dim_in, eps=1e-12)
        self.ffn = FeedForward(dim_in, ffn_hidden_dim)
        self.layer_norm = nn.LayerNorm(dim_in, eps=1e-12)

    def forward(self, x):
        attention = self.self_attention(x)
        attention = x + attention
        attention = self.layer_norm(attention)
        ffn = self.ffn(attention)
        ffn = ffn + attention
        ffn = self.layer_norm(ffn)
        return ffn


if __name__ == '__main__':
    att = SelfAttention(dim_in=136, hidden_dim=256)
    data = torch.randn(32, 17, 136)
    out = att(data)
    print(out.shape)

