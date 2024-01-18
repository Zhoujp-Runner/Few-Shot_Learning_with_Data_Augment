# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 21:31 2023/3/18
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.component import (
    ConcatSquashLinear,
    TransformerLayer,
    AdaEB,
    time_embedding
)
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
                nn.Linear(input_dim + 4, 128),
                nn.ReLU(),
                nn.Linear(128, input_dim)
            ]
        )
        # 时间embedding层
        self.time_embeddings = nn.ModuleList(
            [
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
        # if attribute is None:
        #     for idx, embedding in enumerate(self.time_embeddings):
        #         t_embedding = embedding(t)  # [batch_size, 1, hidden_dim]
        #         t_embedding = t_embedding.squeeze(1)  # [batch_size, hidden_dim]
        #         x = self.linear_layers[2 * idx](x)  # [batch_size, hidden_dim]
        #         x = x + t_embedding  # [batch_size, hidden_dim]
        #         x = self.linear_layers[2 * idx + 1](x)  # [batch_size, hidden_dim]
        # else:
        #     for idx, embedding in enumerate(self.time_embeddings):
        #         t_embedding = embedding(t)  # [batch_size, 1, hidden_dim]
        #         t_embedding = t_embedding.squeeze(1)  # [batch_size, hidden_dim]
        #         att_embedding = self.attribute_embedding_linear[idx](attribute)  # [batch_size, hidden_dim]
        #         # att_embedding = att_embedding.squeeze(1)  # [batch_size, hidden_dim]  使用的是Linear， 没必要添加这一行
        #         x = self.linear_layers[2 * idx](x)  # [batch_size, hidden_dim]
        #         x = x + t_embedding + att_embedding  # [batch_size, hidden_dim]
        #         x = self.linear_layers[2 * idx + 1](x)  # [batch_size, hidden_dim]
        # x = self.linear_layers[-1](x)  # [batch_size, hidden_dim]

        batch_size = x.shape[0]
        if attribute.shape[0] == 1:
            attribute = attribute.expand(batch_size, 4)
        input = torch.cat([x, attribute], dim=-1)
        out = self.linear_layers[0](input)
        t_embedding = self.time_embeddings[0](t)  # [batch_size, 1, hidden_dim]
        t_embedding = t_embedding.squeeze(1)  # [batch_size, hidden_dim]
        out = out + t_embedding
        out = self.linear_layers[1](out)
        out = self.linear_layers[2](out)

        return out  # [batch_size, input_dim]


class ConcatModel(nn.Module):
    def __init__(self, dim_in, dim_condition):
        super(ConcatModel, self).__init__()
        self.concat1 = ConcatSquashLinear(dim_in, 128, dim_condition + 3)
        self.concat2 = ConcatSquashLinear(128, 256, dim_condition + 3)
        # self.concat3 = ConcatSquashLinear(256, 512, dim_condition + 3)
        # self.concat4 = ConcatSquashLinear(512, 256, dim_condition + 3)
        self.concat5 = ConcatSquashLinear(256, 128, dim_condition + 3)
        self.concat6 = ConcatSquashLinear(128, dim_in, dim_condition + 3)
        self.type = "ConcatLinear"

    def forward(self, x, t, attribute):
        """
        :param x: t时刻的输入 [batch_size, dim]
        :param t: 时间t [batch_size, 1]
        :param attribute: 属性 batch_size, attribute_dim]
        """
        batch_size = x.shape[0]
        if attribute.shape[0] == 1:
            attribute = attribute.expand(batch_size, 4)

        time_emb = torch.cat([t, torch.sin(t), torch.cos(t)], dim=-1)  # [batch_size, 3]
        condition_emb = torch.cat([time_emb, attribute], dim=-1)  # [batch_size, attribute_dim + 3]

        out = self.concat1(x, condition_emb)
        out = self.concat2(out, condition_emb)
        # out = self.concat3(out, condition_emb)
        # out = self.concat4(out, condition_emb)
        out = self.concat5(out, condition_emb)
        out = self.concat6(out, condition_emb)

        out = F.leaky_relu(out)
        return out


class AttentionModel(nn.Module):
    def __init__(self, dim_in, dim_condition):
        super(AttentionModel, self).__init__()
        self.concat = ConcatSquashLinear(dim_in, dim_in*2, dim_condition+3)
        self.transformer_encoder = TransformerLayer(dim_in*2, dim_in*4, dim_in*4)
        self.linear = ConcatSquashLinear(dim_in*2, dim_in, dim_condition+3)

        self.type = 'SelfAttentionModel'

    def forward(self, x, t, attribute):

        batch_size = x.shape[0]
        if attribute.shape[0] == 1:
            attribute = attribute.expand(batch_size, 1, 4)
        if len(attribute.shape) == 2:
            attribute = attribute.unsqueeze(1)

        if len(t.shape) == 2:
            t = t.unsqueeze(1)

        time_emb = torch.cat([t, torch.sin(t), torch.cos(t)], dim=-1)  # [batch_size, 1, 3]
        condition_emb = torch.cat([time_emb, attribute], dim=-1)  # [batch_size, 1, attribute_dim + 3]

        concat = self.concat(x, condition_emb)
        context = self.transformer_encoder(concat)
        result = self.linear(context, condition_emb)

        return result


class TestModel(nn.Module):

    def __init__(self, n_steps, num_unit=128):
        """
        初始化类
        :param num_unit: 隐藏层神经元的数量
        """
        super(TestModel, self).__init__()
        # 线性层组合
        self.linears = nn.ModuleList(
            [
                nn.Linear(2, num_unit),
                nn.ReLU(),
                nn.Linear(num_unit, num_unit),
                nn.ReLU(),
                nn.Linear(num_unit, num_unit),
                nn.ReLU(),
                nn.Linear(num_unit, 2)
            ]
        )
        # embedding层组合，不同深度上权重不一样，增加参数但不增加计算量
        self.step_embeddings = nn.ModuleList(
            [
                nn.Embedding(n_steps, num_unit),
                nn.Embedding(n_steps, num_unit),
                nn.Embedding(n_steps, num_unit)
            ]
        )

    def forward(self, x, t):
        """
        :param x: [batch_size, 2]
        :param t: [batch_size, 1]
        :return: [batch_size, 2]
        """
        for idx, embedding in enumerate(self.step_embeddings):
            t_embedding = embedding(t)  # [batch_size, 1, num_unit]
            t_embedding = t_embedding.squeeze(1)  # [batch_size, num_unit]
            x = self.linears[2 * idx](x)  # [batch_size, num_unit]
            x += t_embedding
            x = self.linears[2 * idx + 1](x)
        x = self.linears[-1](x)
        return x


class TorchAttentionModel(nn.Module):
    pass


class AdaModel(nn.Module):
    def __init__(self, dim_in, dim_hidden, attribute_dim, num_steps, dataset):
        super(AdaModel, self).__init__()
        self.emb_layer = AdaEB(dim_hidden, attribute_dim, num_steps, dataset)
        self.in_layer = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
        )
        self.out_layer = nn.Linear(dim_hidden, dim_in)
        self.type = 'AdaModel'

    def forward(self, x, t, att):
        h = self.in_layer(x)
        emb = self.emb_layer(h, t, att)
        out = self.out_layer(emb)
        return out


class GuidedClassifier(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_hidden,
                 dim_out,
                 diffusion_num_step):
        """
        用该分类器的梯度引导DDPM采样
        :param dim_in: 输入维度
        :param dim_hidden: 第一层的输出维度，也是time_embedding的维度
        :param dim_out: 输出维度（即为类别数量）
        :param diffusion_num_step: 扩散步骤
        """
        super(GuidedClassifier, self).__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(dim_in, dim_hidden),
             nn.SiLU(),
             nn.Linear(dim_hidden, 2*dim_hidden),
             nn.SiLU(),
             nn.Linear(2*dim_hidden, 2*dim_hidden),
             nn.SiLU(),
             nn.Linear(2*dim_hidden, dim_hidden),
             nn.SiLU()]
        )
        self.out_layer = nn.Linear(dim_hidden, dim_out)

        # # 使用Embedding层作为嵌入表达
        # self.t_emb = nn.ModuleList(
        #     [nn.Embedding(diffusion_num_step, dim_hidden),
        #      nn.Embedding(diffusion_num_step, 2*dim_hidden),
        #      nn.Embedding(diffusion_num_step, 2*dim_hidden),
        #      nn.Embedding(diffusion_num_step, dim_hidden)]
        # )

        self.time_emb = nn.Sequential(
            nn.Linear(2, dim_hidden),
            nn.SiLU(),
            nn.Linear(dim_hidden, dim_hidden)
        )

        self.emb_layers = nn.ModuleList(
            [nn.SiLU(),
             nn.Linear(dim_hidden, dim_hidden*2),
             nn.SiLU(),
             nn.Linear(dim_hidden, dim_hidden*4),
             nn.SiLU(),
             nn.Linear(dim_hidden, dim_hidden*4),
             nn.SiLU(),
             nn.Linear(dim_hidden, dim_hidden*2)]
        )

    def forward(self, x, time_step):
        # 检查batch_size是否匹配
        # 检查维度数是否匹配
        if len(time_step.shape) < 2:
            time_step = time_step[..., None]

        emb = self.time_emb(time_embedding(time_step))

        for index in range(4):
            x = self.layers[2 * index](x)
            emb_out = self.emb_layers[2 * index](emb)
            emb_out = self.emb_layers[2 * index + 1](emb_out)
            scale, shift = torch.chunk(emb_out, 2, dim=-1)
            x = x * (1 + scale) + shift
            x = self.layers[2 * index + 1](x)

        return self.out_layer(x)


class Generator(nn.Module):
    def __init__(self, latent_dim, out_dim, n_class, emb_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.type = "Generator"
        self.emb = nn.Embedding(n_class, emb_dim)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + emb_dim, int(out_dim / 2)),
            nn.LeakyReLU(),
            nn.Linear(int(out_dim / 2), out_dim)
        )

    def forward(self, z, attribute):
        # print(attribute.shape)
        # print(self.emb(attribute).shape)
        gen_input = torch.cat([z, self.emb(attribute)], dim=-1)
        return self.model(gen_input)


class Discriminator(nn.Module):
    def __init__(self, feature_dim, n_class, emb_dim):
        super(Discriminator, self).__init__()
        self.emb = nn.Embedding(n_class, emb_dim)
        self.model = nn.Sequential(
            nn.Linear(feature_dim + emb_dim, int(feature_dim / 2)),
            nn.ReLU(),
            nn.Linear(int(feature_dim / 2), int(feature_dim / 4)),
            nn.ReLU(),
            nn.Linear(int(feature_dim / 4), 1),
            nn.Sigmoid()
        )

    def forward(self, x, attribute):
        dis_input = torch.cat([x, self.emb(attribute)], dim=-1)
        return self.model(dis_input)


if __name__ == '__main__':
    # model = AttentionModel(64, 4)
    # input = torch.randn(32, 17, 64)
    # attribute = torch.randn(32, 4)
    # if len(attribute.shape) == 2:
    #     attribute = attribute.unsqueeze(1)
    # t = torch.randn(32, 1, 1)
    # out = model(input, t, attribute)
    # print(out.shape)
    # model = AdaModel(64, 128, 4)
    # input = torch.randn(32, 64)
    # attribute = torch.randn(32, 4)
    # t = torch.randn(32, 1)
    # print(model(input, t, attribute).shape)
    model = GuidedClassifier(16, 32, 21, 50)
    input = torch.randn(32, 16)
    t = torch.ones(32).int()
    out = model(input, t)
    print(out.shape)

