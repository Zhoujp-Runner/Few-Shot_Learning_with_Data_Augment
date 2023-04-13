# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 19:56 2023/3/31
"""
本文件的主要作用就是测试
"""
import os
import numpy as np

import yaml
from easydict import EasyDict
import logging

from process_data.dataset import FaultDataset
from models.diffusion import DiffusionModel
from models.model import MLPModel, ConcatModel, AttentionModel
from process_data.augment import DataAugment
from models.classification import TrainClassification


# 训练diffusion model
# 数据增强
# 分别使用未增强后的数据与使用增强后的数据进行测试


def get_all_config(config_path):
    """
    根据路径加载配置参数，返回一个EasyDict类型的文件
    :param config_path: 配置文件路径
    :return: EasyDict()
    """
    with open(config_path) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    # config = EasyDict(config)
    return config


def get_config_list(all_config):
    """
    根据总的配置文件，生成不同的配置文件
    :param all_config: 总的配置文件
    :return: [config0, config1, ...]
    """
    shots_num = all_config["shots_num"]
    methods = all_config["methods"]

    # 对于各个config文件都相同的部分
    config_updated = dict()
    config_updated.update(all_config["data"])
    config_updated.update(all_config["augment"])
    config_updated["diffusion_model_root"] = all_config["diffusion_model"]["root"]
    config_updated["classification_model_root"] = all_config["classification_model"]["root"]
    config_updated.update(all_config["log"])
    config_updated["ways_num"] = all_config["ways_num"]
    config_updated["information"] = all_config["information"]
    config_updated.update(all_config["diffusion_parameters"])
    config_updated.update(all_config["train_parameters"])
    config_updated["augment_num"] = all_config["augment_num"]

    config_list = []
    for shot_num in shots_num:
        for method in methods:
            # 创建当前方法的配置文件
            config = dict()
            config["shots_num"] = shot_num
            config["method"] = method
            config.update(config_updated)
            config_list.append(config)

    return config_list


def get_dataset(config, mode='train', method='LDA', augment=False):
    return FaultDataset(config, mode, method, augment)


def set_log():
    """设置log文件"""
    logger = logging.getLogger("AugmentLog")
    logger.setLevel(logging.DEBUG)

    # 清空该log的句柄
    for handle in logger.handlers:
        logger.removeHandler(handle)

    # self.filehandle = logging.FileHandler(self.config.save_log_path)
    file_path = "experiments\\logs\\test\\test.log"
    filehandle = logging.FileHandler(file_path)
    filehandle.setLevel(logging.DEBUG)

    fmt = "%(message)s"
    formatter = logging.Formatter(fmt)

    filehandle.setFormatter(formatter)
    logger.addHandler(filehandle)

    return logger


if __name__ == '__main__':
    test_config_path = "configs\\test_config.yaml"
    all_config = get_all_config(test_config_path)
    configs = get_config_list(all_config)
    epochs = np.array(range(10 - 1, 2000, 10))
    index = np.argmin(np.abs(epochs - 1534))
    # test_logger = set_log()
    # min_epochs = [1725, 1638, 1649, 1983, 1919, 1934, 1794, 1863, 1999, 1981, 1969, 1914, 1895, 1907, 1741, 1981, 1988, 1996, 1986, 1989]
    for idx, config in enumerate(configs):
        print("=========================Diffusion Training=========================")
        config = EasyDict(config)
        diffusion_model = DiffusionModel(config)

        diffusion_dataset = FaultDataset(config, method=config.method)
        dim_in = diffusion_dataset.train_data.shape[-1]
        dim_condition = diffusion_dataset.train_attribute.shape[-1]
        linear_model = MLPModel(input_dim=dim_in, num_steps=diffusion_model.num_diffusion_steps)
        concat_model = ConcatModel(dim_in=dim_in, dim_condition=dim_condition)
        attention_model = AttentionModel(dim_in=dim_in, dim_condition=dim_condition)

        min_loss, min_epoch = diffusion_model.train(diffusion_dataset, linear_model)
        print("=========================Training done==========================")
        # diffusion_message = f"Diffusion Training : min_epoch = {min_epoch} : min_loss = {min_loss}"
        # test_logger.info(diffusion_message)
        # print(diffusion_message)

        # epochs = np.array(range(config.checkpoint_interval-1, config.epochs, config.checkpoint_interval))
        # index = np.argmin(np.abs(epochs - min_epoch))
        # epoch = epochs[index]

        # # min_epoch = min_epochs[idx]
        print("=========================Augment===========================")
        augment = DataAugment(config, min_epoch)
        augment.data_augment(dim_in=dim_in,
                             dim_condition=dim_condition,
                             model_type=linear_model.type,
                             ways=diffusion_dataset.ways)
        print("=========================Augment done===========================")

        print("=========================Classification===========================")
        classification = TrainClassification(config)
        result = classification.train_loop()

        # # 记录结果
        # message = f"{config.shots_num}_{config.method} classification result:"
        # message_no_augment = f"Without Augment: epoch{result[0][0]}---accuracy={result[0][1]}"
        # message_augment = f"Without Augment: epoch{result[1][0]}---accuracy={result[1][1]}"
        # test_logger.info(message)
        # test_logger.info(message_no_augment)
        # test_logger.info(message_augment)
        print("=========================Classification done===========================")



