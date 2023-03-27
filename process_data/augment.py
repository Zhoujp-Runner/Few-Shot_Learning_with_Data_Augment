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
import numpy as np
from easydict import EasyDict
from itertools import product

from models.diffusion import DiffusionModel
from models.model import MLPModel, ConcatModel


with open("..\\configs\\config_0.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

config = EasyDict(config)


def data_augment(config, model_type):
    """
    使用扩散模型生成新数据，用于数据增强
    :param config:
    :param model_type:
    :return:
    """
    # 生成模型加载路径
    if model_type == "MLP":
        epoch = 499
        batch_size = config.batch_size
        load_name = f'epoch{epoch}_bs{batch_size}_checkpoint.pkl'
        load_root = config.save_model_root_path
        shots_num = f"{config.shots_num}-shots"
        load_path = os.path.join(load_root, model_type, shots_num, load_name)

        # 加载预测模型
        input_dim = 64
        num_steps = config.num_diffusion_steps
        model = MLPModel(input_dim, num_steps)

        model_dict = torch.load(load_path)
        model_state = model_dict['model_state_dict']
        model.load_state_dict(model_state)
    elif model_type == "ConcatLinear":
        epoch = 539
        load_name = f'epoch{epoch}_checkpoint.pkl'
        load_root = config.save_model_root_path
        sub_dir_name = "concat_linear3"
        load_path = os.path.join(load_root, model_type, sub_dir_name, load_name)

        # 加载预测模型
        model = ConcatModel(dim_condition=4)

        model_dict = torch.load(load_path)
        model_state = model_dict['model_state_dict']
        model.load_state_dict(model_state)
    else:
        raise ValueError("There is no such model_type, please try another type!")

    # 加载扩散模型
    diffusion_model = DiffusionModel(config)

    # 每一个属性需要扩增的数据的大小
    augment_size = (config.augment_num, 64)  # TODO 这个64需要改一下，保持和那边降维的数据维度同步

    # 生成所有类别
    attributes = []
    cooler, valve, pump, hydraulic = config.information
    att_iter = product(cooler, valve, pump, hydraulic)
    for item in att_iter:
        attributes.append(list(item))
    attributes = torch.FloatTensor(attributes)

    # 扩增数据集
    augment_data = []
    augment_attribute = []
    for attribute in attributes:
        attribute = attribute.unsqueeze(0)  # [1, 4]
        # TODO 这里对于每一个属性使用同一个diffusion model进行生成样本，是否有问题？
        data = diffusion_model.sample_loop(model, augment_size, attribute=attribute)
        augment_attribute.append(attribute.expand(100, 4).numpy())
        augment_data.append(data.numpy())
    augment_data = np.concatenate(augment_data, axis=0)  # [100*class_num, dim]
    augment_attribute = np.concatenate(augment_attribute, axis=0)  # [100*class_num, 4]

    augment_dict = {
        "data": augment_data,
        "attribute": augment_attribute
    }

    # 保存生成的数据
    save_augment_root = config.save_augment_root
    save_augment_name = f"{model_type}2_augment_num_{config.augment_num}.pkl"
    save_augment_path = os.path.join(save_augment_root, save_augment_name)
    with open(save_augment_path, 'wb') as save_file:
        dill.dump(augment_dict, save_file)


if __name__ == '__main__':
    data_augment(config, model_type='ConcatLinear')
    # root = config.save_augment_root
    # name = f"augment_num_{config.augment_num}.pkl"
    # path = os.path.join(root, name)
    # with open(path, 'rb') as f:
    #     data = dill.load(f)
    # print(data['attribute'][0])
    # print(data['attribute'][100])
    # print(data['attribute'][200])

