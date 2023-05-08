# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 14:56 2023/3/27
"""
1. 使用训练好的模型进行数据增强
"""
import os
import yaml
import dill
import torch
import logging
import numpy as np
from easydict import EasyDict
from itertools import product

from models.diffusion import DiffusionModel
from models.model import MLPModel, ConcatModel, AttentionModel, AdaModel
from process_data.analysis import attribute_standard


class DataAugment(object):
    def __init__(self, config, epoch):
        super(DataAugment, self).__init__()
        self.config = config
        self.epoch = epoch
        self._set_log()

    def _set_log(self):
        """设置log文件"""
        self.logger = logging.getLogger("AugmentLog")
        self.logger.setLevel(logging.DEBUG)

        # 清空该log的句柄
        for handle in self.logger.handlers:
            self.logger.removeHandler(handle)

        # self.filehandle = logging.FileHandler(self.config.save_log_path)
        file_root = self.config.augment_root
        file_name = f"augment_{self.config.shots_num}_{self.config.method}.log"
        file_path = os.path.join(file_root, file_name)
        self.filehandle = logging.FileHandler(file_path)
        self.filehandle.setLevel(logging.DEBUG)

        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self.formatter = logging.Formatter(fmt)

        self.filehandle.setFormatter(self.formatter)
        self.logger.addHandler(self.filehandle)

    def data_augment(self, dim_in, dim_condition, model_type, ways, time=0):
        """
        使用扩散模型生成新数据，用于数据增强
        """
        self.logger.info(f"time: {time}")
        self.logger.info(f"model_type : {model_type}")
        self.logger.info(f"augment_num : {self.config.augment_num}")
        self.logger.info(f"loading epoch : {self.epoch}")
        self.logger.info(f"ways: {ways}")

        # 生成模型加载路径
        if model_type == "MLP":
            load_name = f'epoch{self.epoch}_checkpoint.pkl'
            load_root = self.config.diffusion_model_root
            sub_dir_name = f"{self.config.shots_num}_{self.config.method}"
            load_path = os.path.join(load_root, model_type, sub_dir_name, load_name)

            # 加载预测模型
            num_steps = self.config.num_diffusion_steps
            model = MLPModel(dim_in, num_steps)

            model_dict = torch.load(load_path)
            model_state = model_dict['model_state_dict']
            model.load_state_dict(model_state)

        elif model_type == "ConcatLinear":
            load_name = f'epoch{self.epoch}_checkpoint.pkl'
            load_root = self.config.diffusion_model_root
            sub_dir_name = f"{self.config.shots_num}_{self.config.method}"
            load_path = os.path.join(load_root, model_type, sub_dir_name, load_name)

            # 加载预测模型
            model = ConcatModel(dim_in=dim_in, dim_condition=dim_condition)

            model_dict = torch.load(load_path)
            model_state = model_dict['model_state_dict']
            model.load_state_dict(model_state)

        elif model_type == 'SelfAttentionModel':
            load_name = f'epoch{self.epoch}_checkpoint.pkl'
            load_root = self.config.diffusion_model_root
            sub_dir_name = f"{self.config.shots_num}_{self.config.method}"
            load_path = os.path.join(load_root, model_type, sub_dir_name, load_name)

            # 加载预测模型
            model = AttentionModel(dim_in=dim_in, dim_condition=dim_condition)

            model_dict = torch.load(load_path)
            model_state = model_dict['model_state_dict']
            model.load_state_dict(model_state)

        elif model_type == 'AdaModel':
            load_name = f'epoch{self.epoch}_checkpoint.pkl'
            load_root = self.config.diffusion_model_root
            sub_dir_name = f"{self.config.shots_num}_{self.config.method}"
            load_path = os.path.join(load_root, model_type, sub_dir_name, load_name)

            # 加载预测模型
            model = AdaModel(dim_in=dim_in, dim_hidden=128, attribute_dim=dim_condition, num_steps=self.config.num_diffusion_steps)

            model_dict = torch.load(load_path)
            model_state = model_dict['model_state_dict']
            model.load_state_dict(model_state)
        else:
            raise ValueError("There is no such model_type, please try another type!")

        # 加载扩散模型
        diffusion_model = DiffusionModel(self.config)

        # 每一个属性需要扩增的数据的大小
        if self.config.method == 'Split Standard Dim3 PCA' or self.config.method == 'Split LDA Standard Dim3':
            augment_size = (self.config.augment_num, 17, dim_in)
        else:
            augment_size = (self.config.augment_num, dim_in)

        # # 生成所有类别
        # attributes = []
        # cooler, valve, pump, hydraulic = self.config.information
        # att_iter = product(cooler, valve, pump, hydraulic)
        # for item in att_iter:
        #     attributes.append(list(item))
        # attributes = np.array(attributes)
        # attributes = attribute_standard(attributes, self.config.information)  # 归一化属性
        # attributes = torch.FloatTensor(attributes)
        # # attributes = attributes[:5]  # TODO 只生成前5个类别
        attributes = torch.FloatTensor(ways)  # 需要生成的类别
        self.logger.info(f"Generate Attributes: {attributes}")

        # 扩增数据集
        augment_data = []
        augment_attribute = []
        for attribute in attributes:
            attribute = attribute.unsqueeze(0)  # [1, 4]
            # TODO 这里对于每一个属性使用同一个diffusion model进行生成样本，是否有问题？
            data = diffusion_model.sample_loop(model, augment_size, attribute=attribute)
            augment_attribute.append(attribute.expand(augment_size[0], 4).numpy())
            augment_data.append(data.cpu().numpy())
        augment_data = np.concatenate(augment_data, axis=0)  # [100*class_num, dim]
        augment_attribute = np.concatenate(augment_attribute, axis=0)  # [100*class_num, 4]
        augment_dict = {
            "data": augment_data,
            "attribute": augment_attribute
        }
        # print(diffusion_model.sample_list)

        # 保存生成的数据
        save_augment_root = self.config.augment_data_root
        save_augment_name = f"{self.config.shots_num}_{self.config.method}_{self.config.augment_num}.pkl"
        save_augment_path = os.path.join(save_augment_root, save_augment_name)
        with open(save_augment_path, 'wb') as save_file:
            dill.dump(augment_dict, save_file)


if __name__ == '__main__':
    with open("..\\configs\\config_0.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = EasyDict(config)

    path = "..\\processed_data\\test\\augment\\1_Split Standard Dim3 PCA_10.pkl"
    with open(path, 'rb') as f:
        data = dill.load(f)
    print(data["data"].shape)
    print(data["attribute"].shape)
    # root = config.save_augment_root
    # name = f"augment_num_{config.augment_num}.pkl"
    # path = os.path.join(root, name)
    # with open(path, 'rb') as f:
    #     data = dill.load(f)
    # print(data['attribute'][0])
    # print(data['attribute'][100])
    # print(data['attribute'][200])

