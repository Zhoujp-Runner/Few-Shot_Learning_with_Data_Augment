# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 21:31 2023/3/18
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.component import ConcatSquashLinear
"""
1. 纯全连接神经网络
2. 使用ConcatSquashLinear构建的全连接神经网络
"""


class MLPModel(nn.Module):
    def __init__(self, input_dim, num_steps):
        super(MLPModel, self).__init__()
        # 网络的线性层
        self.linear_layers = nn.ModuleList(
            [
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, input_dim)
            ]
        )
        # 时间embedding层
        self.time_embeddings = nn.ModuleList(
            [
                nn.Embedding(num_steps, 128),
                nn.Embedding(num_steps, 256),
                nn.Embedding(num_steps, 128)
            ]
        )
        # TODO 属性linear层,先不加激活函数
        self.attribute_embedding_linear = nn.ModuleList(
            [
                nn.Linear(4, 128),
                nn.Linear(4, 256),
                nn.Linear(4, 128)
            ]
        )
        # 网络的类型
        self.type = "MLP"

    def forward(self, x, t, attribute=None):
        """
        这里没有考虑好将attribute和哪个做拼接，先单独做一个linear层出来
        后续看一下AC-GAN里是怎么操作的
        :param x: [batch_size, dim]
        :param t: [batch_size, 1]
        :param attribute: [batch_size, attribute_dim] 注意这里的attribute_dim的维度为4，不考虑第五个维度
        :return: [batch_size, dim]
        """
        if attribute is None:
            for idx, embedding in enumerate(self.time_embeddings):
                t_embedding = embedding(t)  # [batch_size, 1, hidden_dim]
                t_embedding = t_embedding.squeeze(1)  # [batch_size, hidden_dim]
                x = self.linear_layers[2 * idx](x)  # [batch_size, hidden_dim]
                x = x + t_embedding  # [batch_size, hidden_dim]
                x = self.linear_layers[2 * idx + 1](x)  # [batch_size, hidden_dim]
        else:
            for idx, embedding in enumerate(self.time_embeddings):
                t_embedding = embedding(t)  # [batch_size, 1, hidden_dim]
                t_embedding = t_embedding.squeeze(1)  # [batch_size, hidden_dim]
                att_embedding = self.attribute_embedding_linear[idx](attribute)  # [batch_size, hidden_dim]
                # att_embedding = att_embedding.squeeze(1)  # [batch_size, hidden_dim]  使用的是Linear， 没必要添加这一行
                x = self.linear_layers[2 * idx](x)  # [batch_size, hidden_dim]
                x = x + t_embedding + att_embedding  # [batch_size, hidden_dim]
                x = self.linear_layers[2 * idx + 1](x)  # [batch_size, hidden_dim]
        x = self.linear_layers[-1](x)  # [batch_size, hidden_dim]
        return x  # [batch_size, input_dim]


class ConcatModel(nn.Module):
    def __init__(self, dim_condition):
        super(ConcatModel, self).__init__()
        self.concat1 = ConcatSquashLinear(64, 128, dim_condition + 3)
        self.concat2 = ConcatSquashLinear(128, 256, dim_condition + 3)
        self.concat3 = ConcatSquashLinear(256, 512, dim_condition + 3)
        self.concat4 = ConcatSquashLinear(512, 256, dim_condition + 3)
        self.concat5 = ConcatSquashLinear(256, 128, dim_condition + 3)
        self.concat6 = ConcatSquashLinear(128, 64, dim_condition + 3)
        self.type = "ConcatLinear"

    def forward(self, x, t, attribute):
        """
        :param x: t时刻的输入 [batch_size, dim]
        :param t: 时间t [batch_size, 1]
        :param attribute: 属性 batch_size, attribute_dim]
        """
        if attribute.shape[0] == 1:
            attribute = attribute.expand(100, 4)

        time_emb = torch.cat([t, torch.sin(t), torch.cos(t)], dim=-1)  # [batch_size, 3]
        condition_emb = torch.cat([time_emb, attribute], dim=-1)  # [batch_size, attribute_dim + 3]

        out = self.concat1(x, condition_emb)
        out = self.concat2(out, condition_emb)
        out = self.concat3(out, condition_emb)
        out = self.concat4(out, condition_emb)
        out = self.concat5(out, condition_emb)
        out = self.concat6(out, condition_emb)

        out = F.leaky_relu(out)
        return out


if __name__ == '__main__':
    m = ConcatModel(4)
    print(m)

