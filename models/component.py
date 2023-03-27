# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 19:04 2023/3/27
"""
一些用于其他网络中的组件
"""

import torch
from torch import nn


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
