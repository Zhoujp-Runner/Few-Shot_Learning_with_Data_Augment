# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 16:47 2023/3/15
"""
1. 加载处理后的数据
2. 划分数据集(先不划分数据集，先将扩散模型搭完后再划分数据)
3. 建立映射关系：__getitem__, __len__
"""
import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import dill
import yaml
from easydict import EasyDict
from itertools import product


with open("..\\configs\\config_0.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

config = EasyDict(config)


class FaultDataset(Dataset):
    def __init__(self,
                 config,
                 mode='train',
                 method='LDA',
                 augment=False,
                 with_test=False):
        """
        :param config: config文件
        :param mode: 'train' or 'test'
        :param method: 'LDA' or 'PCA'
        :param augment: 表示是否使用扩增数据集
        :param with_test: 是否要将测试集包括进生成模型的训练集中
        """
        super(FaultDataset, self).__init__()

        if method == 'LDA':
            self.data_path = config.save_lda_path
        elif method == 'PCA':
            self.data_path = config.save_pca_path
        else:
            raise ValueError("There is no such method for dim_decay!")

        if not os.path.exists(self.data_path):
            raise OSError(f"There is not a existed path. Error path: {self.data_path}")
        with open(self.data_path, 'rb') as f:
            self.source_data = dill.load(f)

        if augment:
            root = config.save_augment_root
            name = f"ConcatLinear_augment_num_{config.dataset_augment_num}.pkl"
            self.augment_path = os.path.join(root, name)
            with open(self.augment_path, 'rb') as aug_file:
                self.augment_dict = dill.load(aug_file)

            self.augment_data = self.augment_dict["data"]
            self.augment_attribute = self.augment_dict["attribute"]

        self.data = self.source_data["data"]
        self.attribute = self.source_data["attribute"]
        self.information = config.information

        self.mode = mode
        self.augment = augment
        self.with_test = with_test
        self.train_data = None
        self.train_attribute = None
        self.test_data = None
        self.test_attribute = None
        self.classes = []

        self.shots_num = config.shots_num
        if self.shots_num > 9:  # 对于当前所用的数据集来说，每一个类别最小的样本数量为10，所以num_shots应该小于等于9
            raise ValueError(
                "shots_num must be equal or less than 9 for hydraulic systems dataset, please modify the config!"
            )
        self.divide_data()

    def divide_data(self):
        """
        将source_data中的数据分成train, test两部分
        将某一个类别的第一个样本划分为测试集， 即测试集中每个类别只有一个样本
        K-shots问题，训练集需要按照K进行划分，如下：
            将某一类别的除了第一个数据的后num个数据划分为训练集
            其中，num的取值为self.shots_num， 在config文件中定义
        如果augment为True，说明此时需要使用扩增的数据，直接将其拼接在整个数据后面
        """
        cooler, valve, pump, hydraulic = self.information
        att_iter = product(cooler, valve, pump, hydraulic)

        test_indices = []
        train_indices = []
        # 提取出每一类的第一个样本的索引值
        # TODO 提取方法有待改进，两个循环的暴力搜索的时间复杂度有点高
        # TODO 这样子切割使得训练集与测试集完全固定，被筛选掉的样本永远也不会参与计算，可以改成每次创建dataset都是不同的
        for item in att_iter:  # 遍历每一个类别
            num = 0  # 用来计数提取的属性个数
            self.classes.append(list(item))
            for idx, att in enumerate(self.attribute):  # 遍历每一个属性
                att = tuple(att)
                if num == self.shots_num + 1:  # 如果已经提取完对应的样本数量就退出循环
                    break
                if item == att and num != 0:  # 提取完第一个样本后
                    train_indices.append(idx)
                    num += 1
                elif item == att and num == 0:  # 未提取第一个样本
                    test_indices.append(idx)
                    num += 1

        # 拷贝，防止由于test的改变而导致原数据改变
        data = np.copy(self.data)
        attribute = np.copy(self.attribute)
        self.test_data = data[test_indices]
        self.test_attribute = attribute[test_indices]
        self.train_data = data[train_indices]
        self.train_attribute = attribute[train_indices]
        # self.train_data = np.delete(data, test_indices, axis=0)
        # self.train_attribute = np.delete(attribute, test_indices, axis=0)

        # 是否使用扩增的数据
        if self.augment:
            self.train_data = np.concatenate([self.train_data, self.augment_data], axis=0)
            self.train_attribute = np.concatenate([self.train_attribute, self.augment_attribute], axis=0)

        # np.ndarray -> torch.FloatTensor
        self.data = torch.FloatTensor(self.data)
        self.train_data = torch.FloatTensor(self.train_data)
        self.train_attribute = torch.FloatTensor(self.train_attribute)
        self.test_data = torch.FloatTensor(self.test_data)
        self.test_attribute = torch.FloatTensor(self.test_attribute)

    @property
    def _get_len(self):
        if self.mode == 'test':
            return len(self.test_data)
        elif self.mode == 'train' and self.with_test:
            return len(self.data)
        elif self.mode == 'train' and not self.with_test:
            return len(self.train_data)
        else:
            raise ValueError(f'{self.mode} is not set !')

    def __getitem__(self, item):
        if self.mode == 'test':
            return self.test_data[item], self.test_attribute[item]
        elif self.mode == 'train' and self.with_test:
            return self.data[item], self.attribute[item]
        elif self.mode == 'train' and not self.with_test:
            return self.train_data[item], self.train_attribute[item]

    def __len__(self):
        return self._get_len


if __name__ == '__main__':
    dataset = FaultDataset(config, augment=True)
    # print(dataset.__getitem__(1))
    # print(dataset.__len__())
    # print(dataset.test_data)
    # print(dataset.test_attribute)
    print(dataset.train_data.shape)
    print(dataset.train_attribute.shape)
    # print(dataset.test_data == dataset.train_data)
    # print(dataset.classes)
    # indices = [1, 2, 3, 5]
    # test = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]
    # test = np.array(test)
    # dele = np.delete(test, indices, axis=0)
    # dele[0][0] = 0
    # print(test)
    # print(dele)
