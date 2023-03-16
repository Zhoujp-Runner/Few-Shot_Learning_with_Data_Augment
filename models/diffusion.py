# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 17:34 2023/3/15
"""
1. MLP模型
2. 扩散模型的类
    其中包括的功能：
    1）根据给定的参数获得beta
    2）根据beta计算出alpha_bar等一系列参数
    3）扩散过程
"""
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


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

    def forward(self, x, t, attribute):
        """
        这里没有考虑好将attribute和哪个做拼接，先单独做一个linear层出来
        后续看一下AC-GAN里是怎么操作的
        :param x: [batch_size, dim]
        :param t: [batch_size, 1]
        :param attribute: [batch_size. attribute_dim] 注意这里的attribute_dim的维度为4，不考虑第五个维度
        :return: [batch_size, dim]
        """
        for idx, embedding in enumerate(self.time_embeddings):
            t_embedding = embedding(t)  # [batch_size, 1, hidden_dim]
            t_embedding = t_embedding.squeeze(1)  # [batch_size, hidden_dim]
            att_embedding = self.attribute_embedding_linear[idx](attribute)  # [batch_size, 1, hidden_dim]
            att_embedding = att_embedding.squeeze(1)  # [batch_size, hidden_dim]
            x = self.linear_layers[2 * idx](x)  # [batch_size, hidden_dim]
            x = x + t_embedding + att_embedding  # [batch_size, hidden_dim]
            x = self.linear_layers[2 * idx + 1](x)  # [batch_size, hidden_dim]
        x = self.linear_layers[-1](x)  # [batch_size, hidden_dim]
        return x  # [batch_size, hidden_dim]


class DiffusionModel(object):
    def __init__(self,
                 num_diffusion_steps,
                 beta_start=0.0001,
                 beta_end=0.02,
                 attribute=None):
        super(DiffusionModel, self).__init__()
        self.num_diffusion_steps = num_diffusion_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.attribute = attribute  # 数据的属性值

        self._get_bata_schedule()
        self._get_parameters_related_to_alpha_bar()

    def _get_bata_schedule(self):
        """
        通过给定的扩散步骤，得到beta_schedule
        """
        scale = 1000 / self.num_diffusion_steps
        beta_start = scale * self.beta_start
        beta_end = scale * self.beta_end
        self.betas = np.linspace(
            beta_start, beta_end, self.num_diffusion_steps, dtype=np.float64
        )
        print(self.betas)
        print(len(self.betas.shape))

    def _get_parameters_related_to_alpha_bar(self):
        """
        根据beta计算alpha_bar有关的参数
        """
        if not len(self.betas.shape) == 1:
            raise ValueError("betas must be 1-D!")
        self.alphas = 1.0 - self.betas
        self.sqrt_alphas = np.sqrt(self.alphas)
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod_minus_one = np.sqrt(1.0 / self.alphas_cumprod - 1.0)

    def sample_from_posterior(self,
                              model: MLPModel,
                              x_t,
                              t):
        """
        根据后验分布，以及t时刻神经网络模型p的预测值，得到t-1时刻的数据分布并采样
        即 q(x_t-1 | x_t, x_0)
        :param model: 神经网络模型
        :param x_t: 第t时刻的数据 [batch_size, dim]
        :param t: 时间步 [batch_size, 1]
        :return:
        """
        # 高斯噪声采样
        noise = torch.randn_like(x_t)

        # 模型输出
        model_out = model(x_t, t, self.attribute)

        # 计算均值
        coefficient1 = 1.0 / torch.from_numpy(self.sqrt_alphas)[t].float()
        coefficient2 = \
            torch.from_numpy(self.betas)[t].float() / \
            torch.from_numpy(self.sqrt_one_minus_alphas_cumprod)[t].float()
        mean = coefficient1 * (x_t - coefficient2 * model_out)

        # 计算方差
        variance = \
            (1.0 - torch.from_numpy(self.alphas_cumprod_prev)[t].float()) * \
            torch.from_numpy(self.betas)[t].float() / \
            (1.0 - torch.from_numpy(self.alphas_cumprod)[t].float())

        return mean + torch.sqrt(variance) * noise

    def sample_loop(self,
                    model: MLPModel,
                    shape):
        """
        循环采样，逆扩散过程
        x_t -> x_t-1 -> x_t-2 -> ... -> x_0
        :param model: 神经网络模型
        :param shape: 待生成的样本的形状
        :return: 整个逆扩散过程的采样结果的list
        """
        # 生成最初的噪声
        x_t = torch.randn(shape)

        # 反时间步 [T, T-1, T-2, ..., 0]
        time_steps = list(range(self.num_diffusion_steps))[::-1]
        time_steps = tqdm(time_steps)

        # 用于存放每一步的采样结果
        x_sample = [x_t]

        # 进行循环采样，当前步的输出作为下一步的输入
        for t in time_steps:
            t = torch.tensor([t] * shape[0])
            with torch.no_grad():
                x_t_minus_1 = self.sample_from_posterior(model,
                                                         x_t,
                                                         t)
                x_sample.append(x_t_minus_1)
                x_t = x_t_minus_1

        return x_t

    def diffusion_at_time_t(self, x_0: torch.Tensor, t):
        """
        扩散过程，获得第t步的数据
        即q(x_t | x_0)
        :param x_0: 初始数据
        :param t: 时间步t
        :return: 第t步的数据
        """
        noise = torch.randn_like(x_0)

        # 计算均值与标准差
        mean = torch.from_numpy(self.sqrt_alphas_cumprod)[t].float() * x_0
        standard_deviation = torch.from_numpy(self.sqrt_one_minus_alphas_cumprod)[t].float()

        return mean + standard_deviation * noise


if __name__ == '__main__':
    # input = torch.Tensor(100, 64)
    # time = torch.ones((100, 1), dtype=torch.int)
    # att = torch.ones((100, 4), dtype=torch.float)
    # model = MLPModel(64, 1000)
    # x = model(input, time, att)
    # print(x.shape)
    # x = torch.rand((64, 64))
    # dif = DiffusionModel(num_diffusion_steps=100)
    # print(dif.diffusion_at_time_t(x, 10).shape)
    indi = list(range(100))[::-1]
    print(indi)
