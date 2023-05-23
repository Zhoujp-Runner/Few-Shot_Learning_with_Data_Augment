# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 19:56 2023/3/31
"""
本文件的主要作用就是测试
"""
import os
import numpy as np
import torch
import torch.nn.functional as F

import yaml
from easydict import EasyDict
import logging

from process_data.dataset import FaultDataset, TEPDataset
from models.diffusion import DiffusionModel
from models.model import MLPModel, ConcatModel, AttentionModel, AdaModel, GuidedClassifier
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
    config_updated.update(all_config["classifier"])
    config_updated["ways_num"] = all_config["ways_num"]
    config_updated["information"] = all_config["information"]
    config_updated.update(all_config["diffusion_parameters"])
    config_updated.update(all_config["train_parameters"])
    config_updated["augment_num"] = all_config["augment_num"]
    config_updated["dataset_type"] = all_config["dataset_type"]

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
    logger = logging.getLogger("TestLog")
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


def get_classifier(config):
    """加载classifier模型"""
    # 参数要与classifier_train中的参数一致
    classifier = GuidedClassifier(dim_in=16,
                                  dim_hidden=256,
                                  dim_out=21,
                                  diffusion_num_step=50)

    path = config["classifier_path"]
    classifier_checkpoint = torch.load(path)
    classifier_state_dict = classifier_checkpoint["model"]
    classifier.load_state_dict(classifier_state_dict)
    return classifier


if __name__ == '__main__':
    test_config_path = "configs\\test_config.yaml"
    all_config = get_all_config(test_config_path)
    configs = get_config_list(all_config)
    test_times = 10
    epochs = np.array(range(10 - 1, 2000, 10))
    index = np.argmin(np.abs(epochs - 1534))
    test_logger = set_log()
    test_logger.info(f"total_time: {test_times}")

    classifier = get_classifier(configs[0])
    scale = configs[0]["classifier_scale"]

    def condition_func(x_t, t, y=None):
        assert y is not None
        with torch.enable_grad():
            x_in = x_t.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return torch.autograd.grad(selected.sum(), x_in)[0] * scale

    for idx, config in enumerate(configs):

        # 准确率之和
        total_augment_accuracy = 0
        total_source_accuracy = 0
        # 最小的准确率
        min_augment_accuracy = 2
        min_source_accuracy = 2
        # 对应最小准确率的ways
        min_augment_ways = None
        min_source_ways = None
        # 对应最小准确率的次数
        min_source_time = 0
        min_augment_time = 0

        for time in range(test_times):
            if config["dataset_type"] == 'Hydraulic':
                print("=========================Diffusion Training=========================")
                config = EasyDict(config)
                diffusion_model = DiffusionModel(config)

                diffusion_dataset = FaultDataset(config,
                                                 method=config.method,
                                                 use_random_combination=True)
                dim_in = diffusion_dataset.train_data.shape[-1]
                dim_condition = diffusion_dataset.train_attribute.shape[-1]
                linear_model = MLPModel(input_dim=dim_in,
                                        num_steps=diffusion_model.num_diffusion_steps)
                concat_model = ConcatModel(dim_in=dim_in,
                                           dim_condition=dim_condition)
                attention_model = AttentionModel(dim_in=dim_in,
                                                 dim_condition=dim_condition)
                ada_model = AdaModel(dim_in=dim_in,
                                     dim_hidden=128,
                                     attribute_dim=dim_condition,
                                     num_steps=diffusion_model.num_diffusion_steps,
                                     dataset='Hydraulic')

                min_loss, min_epoch = diffusion_model.train(diffusion_dataset, ada_model, time)
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
                                     model_type=ada_model.type,
                                     ways=diffusion_dataset.ways,
                                     time=time,
                                     dataset='Hydraulic',
                                     guided_fn=condition_func)
                print("=========================Augment done===========================")

                print("=========================Classification===========================")
                classification = TrainClassification(config,
                                                     ways=diffusion_dataset.ways)
                result = classification.train_loop(time)
                if min_augment_accuracy > result[1][1]:
                    min_augment_accuracy = result[1][1]
                    min_augment_ways = diffusion_dataset.ways
                    min_augment_time = time
                if min_source_accuracy > result[0][1]:
                    min_source_accuracy = result[0][1]
                    min_source_ways = diffusion_dataset.ways
                    min_source_time = time

                total_augment_accuracy += result[1][1]
                total_source_accuracy += result[0][1]

                # # 记录结果
                # message = f"{config.shots_num}_{config.method} classification result:"
                # message_no_augment = f"Without Augment: epoch{result[0][0]}---accuracy={result[0][1]}"
                # message_augment = f"Without Augment: epoch{result[1][0]}---accuracy={result[1][1]}"
                # test_logger.info(message)
                # test_logger.info(message_no_augment)
                # test_logger.info(message_augment)
                print("=========================Classification done===========================")
            elif config["dataset_type"] == 'TEP':
                print("=========================Diffusion Training=========================")
                config = EasyDict(config)
                diffusion_model = DiffusionModel(config)

                diffusion_dataset = TEPDataset(config,
                                               mode='train')
                # TEP数据集每个样本的数据是52维且类别维度是1
                dim_in = diffusion_dataset.train_data.shape[-1] - 1
                # 要求将1维的类别信息编码成dim_condition维的向量
                dim_condition = 32

                ada_model = AdaModel(dim_in=dim_in,
                                     dim_hidden=64,
                                     attribute_dim=dim_condition,
                                     num_steps=diffusion_model.num_diffusion_steps,
                                     dataset='TEP')

                min_loss, min_epoch = diffusion_model.train(diffusion_dataset, ada_model, time)
                print("=========================Training done==========================")

                print("=========================Augment===========================")
                augment = DataAugment(config, min_epoch)
                augment.data_augment(dim_in=dim_in,
                                     dim_condition=dim_condition,
                                     model_type=ada_model.type,
                                     ways=diffusion_dataset.ways,
                                     time=time,
                                     dataset='TEP',
                                     guided_fn=condition_func)
                print("=========================Augment done===========================")

                print("=========================Classification===========================")
                classification = TrainClassification(config,
                                                     ways=diffusion_dataset.ways)
                result = classification.train_loop(time, data_set=diffusion_dataset)
                if min_augment_accuracy > result[1][1]:
                    min_augment_accuracy = result[1][1]
                    min_augment_ways = diffusion_dataset.ways
                    min_augment_time = time
                if min_source_accuracy > result[0][1]:
                    min_source_accuracy = result[0][1]
                    min_source_ways = diffusion_dataset.ways
                    min_source_time = time

                total_augment_accuracy += result[1][1]
                total_source_accuracy += result[0][1]
                print("=========================Classification done===========================")

        average_augment_accuracy = total_augment_accuracy / test_times
        average_source_accuracy = total_source_accuracy / test_times
        print(f"augment_num: {config.augment_num}")
        print(f"test_times: {test_times}")
        print(f"average_source_accuracy: {average_source_accuracy}")
        print(f"min_source_accuracy: {min_source_accuracy}")
        print(f"min_source_ways: {min_source_ways}")
        print(f"average_augment_accuracy: {average_augment_accuracy}")
        print(f"min_augment_accuracy: {min_augment_accuracy}")
        print(f"min_augment_ways: {min_augment_ways}")

        test_logger.info(f"------------------Average Result------------------")
        test_logger.info(f"shots_num: {config.shots_num}")
        test_logger.info(f"ways_num: {config.ways_num}")
        test_logger.info(f"method: {config.method}")
        test_logger.info(f"augment_num: {config.augment_num}")
        test_logger.info(f"test_times: {test_times}")
        test_logger.info(f"average_source_accuracy: {average_source_accuracy}")
        test_logger.info(f"min_source_accuracy: {min_source_accuracy}")
        test_logger.info(f"min_source_ways: {min_source_ways}")
        test_logger.info(f"average_augment_accuracy: {average_augment_accuracy}")
        test_logger.info(f"min_augment_accuracy: {min_augment_accuracy}")
        test_logger.info(f"min_augment_ways: {min_augment_ways}")
        test_logger.info(f"------------------Average Result------------------")



