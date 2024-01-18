# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 14:53 2024/1/12

import os
import numpy as np
import torch
import logging
from torch.utils.data import DataLoader
from torch.nn import Module
from tqdm import tqdm


def makedir(path):
    """
    如果路径不存在，创建文件夹
    :param path: 文件夹路径
    """
    if not os.path.exists(path):
        os.makedirs(path)


class GanModel(object):
    def __init__(self, config):
        super(GanModel, self).__init__()
        self.config = config
        self.lr = float(config.learning_rate)
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.device = config.device
        self.checkpoint_interval = config.checkpoint_interval

        self._set_log()

    def _set_log(self):
        """设置log文件"""
        self.logger = logging.getLogger("GanLog")
        self.logger.setLevel(logging.DEBUG)

        # 清空log句柄
        for handle in self.logger.handlers:
            self.logger.removeHandler(handle)

        file_root = self.config.gan_root
        if self.config.dataset_type == 'Hydraulic':
            file_name = f"gan_{self.config.shots_num}_{self.config.method}.log"
        elif self.config.dataset_type == 'TEP':
            file_name = f"gan_{self.config.shots_num}_{self.config.dataset_type}.log"
        else:
            raise ValueError('Please use the right dataset!')
        file_path = os.path.join(file_root, file_name)
        self.filehandle = logging.FileHandler(file_path)
        self.filehandle.setLevel(logging.DEBUG)

        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self.formatter = logging.Formatter(fmt)

        self.filehandle.setFormatter(self.formatter)
        self.logger.addHandler(self.filehandle)

    def train(self,
              dataset,
              generator: Module,
              discriminator: Module,
              time=0):
        self.logger.info(
            "===========================================Gan Training========================================"
        )
        self.logger.info(f"time: {time}")
        self.logger.info(f"Dataset: {dataset.dataset_type}")
        self.logger.info(generator)
        self.logger.info(discriminator)
        self.logger.info(f"Batch_size: {self.batch_size}")
        self.logger.info(f"learning_rate: {self.lr}")
        self.logger.info(f"K-shots: k={self.config.shots_num}")
        self.logger.info(f"epochs: {self.epochs}")
        if dataset.dataset_type == 'Hydraulic':
            self.logger.info(f"dataset: method={dataset.method}")
        self.logger.info(f"ways: {dataset.ways}")

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        generator = generator.to(self.device)
        discriminator = discriminator.to(self.device)

        # 隐变量维度
        latent_dim = generator.latent_dim

        adversarial_loss = torch.nn.BCELoss()

        g_optimizer = torch.optim.Adam(generator.parameters(), lr=self.lr)
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=self.lr)

        min_loss = 100
        min_epoch = 0

        for epoch in range(self.epochs):
            cur_loss = 0
            pb_dataloader = tqdm(dataloader, desc=f"Epoch {epoch}: ")
            for idx, (x, attribute) in enumerate(pb_dataloader):
                x = x.to(self.device)
                attribute = attribute.to(self.device)

                # ground truth
                valid_gt = torch.ones(x.shape[0], 1).to(self.device)
                fake_gt = torch.zeros(x.shape[0], 1).to(self.device)

                # 更新generator
                g_optimizer.zero_grad()
                z = torch.randn(x.shape[0], latent_dim).to(self.device)
                gen_data = generator(z, attribute)
                g_loss = adversarial_loss(discriminator(gen_data, attribute), valid_gt)
                g_loss.backward()
                g_optimizer.step()
                cur_loss += g_loss.detach().item()

                # 更新discriminator
                d_optimizer.zero_grad()
                real_out = discriminator(x, attribute)
                fake_out = discriminator(gen_data.detach(), attribute)
                d_loss_1 = adversarial_loss(real_out, valid_gt)
                d_loss_2 = adversarial_loss(fake_out, fake_gt)
                d_loss = (d_loss_1 + d_loss_2) / 2
                d_loss.backward()
                d_optimizer.step()
                # cur_loss += d_loss.detach().item()

                # 输出log
                self.logger.info(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, self.epochs, idx, len(dataloader), d_loss.item(), g_loss.item())
                )
            cur_loss = cur_loss / len(dataloader)
            # 只记录小loss的
            if cur_loss < min_loss:
                min_loss = cur_loss
                min_epoch = epoch
                checkpoint = {
                    "generator_state_dict": generator.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "g_optimizer_state_dict": d_optimizer.state_dict(),
                    "d_optimizer_state_dict": g_optimizer.state_dict()
                }

                sub_dir_name = f"{self.config.shots_num}_{self.config.method}"
                model_name = f"epoch{epoch}_checkpoint.pkl"
                dir_path = os.path.join(self.config.gan_model_root, sub_dir_name)
                makedir(dir_path)
                save_model_path = os.path.join(dir_path, model_name)

                torch.save(checkpoint, save_model_path)

            # # 每隔checkpoint_interval个周期保存一次模型
            # if (epoch+1) % self.checkpoint_interval == 0:
            #
            #     checkpoint = {
            #         "generator_state_dict": generator.state_dict(),
            #         "discriminator_state_dict": discriminator.state_dict(),
            #         "g_optimizer_state_dict": d_optimizer.state_dict(),
            #         "d_optimizer_state_dict": g_optimizer.state_dict()
            #     }
            #
            #     sub_dir_name = f"{self.config.shots_num}_{self.config.method}"
            #     model_name = f"epoch{epoch}_checkpoint.pkl"
            #     dir_path = os.path.join(self.config.gan_model_root, sub_dir_name)
            #     makedir(dir_path)
            #     save_model_path = os.path.join(dir_path, model_name)
            #
            #     torch.save(checkpoint, save_model_path)

        return min_epoch, min_loss

    def sample_loop(self,
                    generator,
                    data_size,
                    attribute,
                    guided_fn=None):
        if attribute.shape[0] != data_size[0]:
            dim = attribute.shape[-1]
            try:
                attribute = attribute.expand(data_size[0])
            except Exception:
                raise ValueError("Attribute shape does not match the data size!")
        with torch.no_grad():
            latent_dim = generator.latent_dim
            z = torch.randn(data_size[0], latent_dim).to(self.device)
            attribute = attribute.to(self.device)
            generator = generator.to(self.device)
            data = generator(z, attribute)
        return data


if __name__ == '__main__':
    o = torch.ones(10, 1, requires_grad=True)
    print(o.requires_grad)
    e = torch.zeros(1, 10)
    print(torch.cat([o, e], dim=-1))

