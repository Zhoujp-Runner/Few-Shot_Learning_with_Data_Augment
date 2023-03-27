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
import os.path
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from process_data.dataset import FaultDataset
from models.model import MLPModel, ConcatModel
import yaml
from easydict import EasyDict
import logging


with open("..\\configs\\config_0.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

config = EasyDict(config)


class DiffusionModel(object):
    def __init__(self, config):
        super(DiffusionModel, self).__init__()
        self.config = config
        self.num_diffusion_steps = config.num_diffusion_steps
        self.beta_start = config.beta_start
        self.beta_end = config.beta_end

        self._get_bata_schedule()
        self._get_parameters_related_to_alpha_bar()
        self._set_log()

        self.checkpoint_interval = config.checkpoint_interval
        self.device = config.device
        self.batch_size = config.batch_size
        self.lr = float(config.learning_rate)
        self.epochs = config.epochs

        self.sample_list = None  # 用于存放逆扩散过程每一步的采样结果

    def _set_log(self):
        """设置log文件"""
        self.logger = logging.getLogger("DiffusionLog")
        self.logger.setLevel(logging.DEBUG)

        self.filehandle = logging.FileHandler(self.config.save_log_path)
        self.filehandle.setLevel(logging.DEBUG)

        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self.formatter = logging.Formatter(fmt)

        self.filehandle.setFormatter(self.formatter)
        self.logger.addHandler(self.filehandle)

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
                              t,
                              attribute=None):
        """
        根据后验分布，以及t时刻神经网络模型p的预测值，得到t-1时刻的数据分布并采样
        即 q(x_t-1 | x_t, x_0)
        :param model: 神经网络模型
        :param x_t: 第t时刻的数据 [batch_size, dim]
        :param t: 时间步 [batch_size, 1]
        :param attribute: 属性矩阵
        :return:
        """
        # 高斯噪声采样
        noise = torch.randn_like(x_t)

        # 模型输出
        model_out = model(x_t, t, attribute)

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
                    shape,
                    attribute=None):
        """
        循环采样，逆扩散过程
        x_t -> x_t-1 -> x_t-2 -> ... -> x_0
        :param model: 神经网络模型
        :param shape: 待生成的样本的形状
        :return: 生成的样本，即x_0
        """
        # 生成最初的噪声
        x_t = torch.randn(shape)

        # 反时间步 [T, T-1, T-2, ..., 0]
        time_steps = list(range(self.num_diffusion_steps))[::-1]
        time_steps = tqdm(time_steps, desc=f"attribute{attribute}:")

        # 用于存放每一步的采样结果
        self.sample_list = [x_t]

        # 进行循环采样，当前步的输出作为下一步的输入
        for t in time_steps:
            t = torch.tensor([t] * shape[0])  # [batch_size]
            t = t.unsqueeze(1)  # [batch_size, 1] 这里是为了采样的系数矩阵的维度符合广播机制
            with torch.no_grad():
                x_t_minus_1 = self.sample_from_posterior(model,
                                                         x_t,
                                                         t,
                                                         attribute)
                self.sample_list.append(x_t_minus_1)
                x_t = x_t_minus_1

        return x_t

    def diffusion_at_time_t(self,
                            x_0: torch.Tensor,
                            t,
                            noise):
        """
        扩散过程，获得第t步的数据
        即q(x_t | x_0)
        :param x_0: 初始数据
        :param t: 时间步t
        :param noise: 噪声
        :return: 第t步的数据
        """

        # 计算均值与标准差
        mean_coefficient = torch.from_numpy(self.sqrt_alphas_cumprod)[t].float().to(self.device)
        mean = mean_coefficient * x_0
        standard_deviation = torch.from_numpy(self.sqrt_one_minus_alphas_cumprod)[t].float().to(self.device)

        return (mean + standard_deviation * noise).float()

    def loss_fn(self,
                model,
                x_0: torch.Tensor,
                attribute=None):
        """目标函数"""
        # 对t进行采样
        batch_size = x_0.shape[0]
        t = self.sample_t(batch_size)

        # 通过扩散过程生成t时刻的数据
        noise = torch.randn_like(x_0)
        x_t = self.diffusion_at_time_t(x_0, t, noise)

        # 模型前向传播
        model_out = model(x_t, t, attribute)

        return torch.mean(torch.square((noise - model_out)))

    def sample_t(self, batch_size):
        """对时间步t进行均匀采样"""
        weight = np.ones([self.num_diffusion_steps])  # 由于均匀采样，所以每个时间步的权重都为1
        p = weight / np.sum(weight)  # 计算每个时间步的采样概率
        time_sample_np = np.random.choice(self.num_diffusion_steps, size=(batch_size,), p=p)  # 以概率p对时间步采样
        time_sample = torch.from_numpy(time_sample_np).long()  # [batch_size]
        time_sample = time_sample.unsqueeze(-1).to(self.device)  # [batch_size, 1]
        return time_sample

    def train(self,
              dataset,
              model):
        """训练模型"""
        # 记录模型的结构以及一些参数
        self.logger.info(model)
        self.logger.info(f"Batch_size: {self.batch_size}")
        self.logger.info(f"diffusion_step: {self.num_diffusion_steps}")
        self.logger.info(f"learning_rate: {self.lr}")
        self.logger.info(f"K-shots: k={config.shots_num}")
        self.logger.info(f"epochs: {self.epochs}")

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            loss_total = 0
            pb_dataloader = tqdm(dataloader, desc=f"Epoch {epoch}: ")
            for x_0, attribute in pb_dataloader:
                x_0 = x_0.to(self.device)
                attribute = attribute.to(self.device)

                loss = self.loss_fn(model, x_0, attribute)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pb_dataloader.set_postfix_str(f"Loss = {loss.item()}")
                loss_total += loss.item()

            loss_mean = loss_total / len(dataloader)
            log_message = f"Epoch{epoch}: Loss = {loss_mean}"
            self.logger.info(log_message)

            # 每隔10个周期保存一次模型
            if (epoch+1) % self.checkpoint_interval == 0:
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                }
                sub_dir_name = "concat_linear3"
                model_name = f"epoch{epoch}_checkpoint.pkl"
                save_model_path = os.path.join(config.save_model_root_path, model.type, sub_dir_name, model_name)
                torch.save(checkpoint, save_model_path)


if __name__ == '__main__':
    # input = torch.Tensor(100, 64)
    # time = torch.ones((100, 1), dtype=torch.int)
    # att = torch.ones((100, 4), dtype=torch.float)
    # model = MLPModel(64, 1000)
    # x = model(input, time, att)
    # print(x.shape)
    # x = torch.rand((64, 64))
    dif = DiffusionModel(config)
    # print(dif.diffusion_at_time_t(x, 10).shape)
    dataset = FaultDataset(config)
    model = MLPModel(64, 3000)
    concat_model = ConcatModel(dim_condition=4)
    dif.train(dataset, concat_model)

    # indi = list(range(100))[::-1]
    # print(indi)
