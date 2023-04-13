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
from sklearn.preprocessing import MinMaxScaler

from process_data.analysis import attribute_standard


class FaultDataset(Dataset):
    def __init__(self,
                 config,
                 mode='train',
                 method='LDA',
                 augment=False,
                 with_train=True,
                 with_test=False):
        """
        :param config: 配置参数
        :param mode: 'train' or 'test'
        :param method: 'LDA' or 'PCA' or 'Standard PCA' or \
                        'Split Standard PCA' or 'Split Standard Dim3 PCA' \
                        or 'LDA Standard' or 'Split LDA Standard' or 'Split LDA Standard Dim3'
        :param augment: 表示是否使用扩增数据集
        :param with_train: 扩增数据集是否包含原训练数据
        :param with_test: 是否要将测试集包括进生成模型的训练集中
        """
        super(FaultDataset, self).__init__()

        if method == 'LDA':
            self.data_path = config.save_lda_path
        elif method == 'PCA':
            self.data_path = config.save_pca_path
        elif method == 'Standard PCA':
            self.data_path = config.save_standard_pca_path
        elif method == 'Split Standard PCA':
            self.data_path = config.save_split_standard_pca_path
        elif method == 'Split Standard Dim3 PCA':
            self.data_path = config.save_split_standard_pca_dim3_path
        elif method == 'LDA Standard':
            self.data_path = config.save_lda_standard_path
        elif method == 'Split LDA Standard':
            self.data_path = config.save_split_lda_standard_path
        elif method == 'Split LDA Standard Dim3':
            self.data_path = config.save_split_lda_standard_dim3_path
        else:
            raise ValueError("There is no such method for dim_decay!")

        if not os.path.exists(self.data_path):
            raise OSError(f"There is not a existed path. Error path: {self.data_path}")
        with open(self.data_path, 'rb') as f:
            self.source_data = dill.load(f)

        if augment:
            # 加载路径
            # if method == 'LDA':
            #     self.augment_path = config.lda
            # elif method == 'PCA':
            #     self.augment_path = config.pca
            # elif method == 'Standard PCA':
            #     self.augment_path = config.standard_pca
            # elif method == 'Split Standard PCA':
            #     self.augment_path = config.split_standard_pca
            # else:
            #     raise ValueError("There is no such method for augment!")
            augment_data_root = config.augment_data_root
            augment_data_name = f"{config.shots_num}_{config.method}_{config.augment_num}.pkl"
            self.augment_path = os.path.join(augment_data_root, augment_data_name)
            with open(self.augment_path, 'rb') as aug_file:
                self.augment_dict = dill.load(aug_file)

            # model = MinMaxScaler()
            # self.augment_data = model.fit_transform(self.augment_dict["data"])
            self.augment_data = self.augment_dict["data"]
            self.augment_attribute = self.augment_dict["attribute"]

        self.data = self.source_data["data"]
        self.attribute = self.source_data["attribute"]
        self.information = config.information

        self.config = config
        self.mode = mode
        self.augment = augment
        self.method = method
        self.with_train = with_train
        self.with_test = with_test
        self.ways = None
        self.train_data = None
        self.train_attribute = None
        self.test_data = None
        self.test_attribute = None
        self.classes = []

        self.ways_num = config.ways_num
        self.shots_num = config.shots_num
        # if self.shots_num > 9:  # 对于当前所用的数据集来说，每一个类别最小的样本数量为10，所以num_shots应该小于等于9
        #     raise ValueError(
        #         "shots_num must be equal or less than 9 for hydraulic systems dataset, please modify the config!"
        #     )
        self.divide_data()

    def divide_data(self):
        """
        将source_data中的数据分成train, test两部分
        将某一个类别的前num-k个样本划分为测试集， 即测试集中每个类别只有k个样本作为训练集，其余样本作为测试集
        K-shots问题，训练集需要按照K进行划分，如下：
            将某一类别前k个数据划分为训练集
            其中，k的取值为self.shots_num， 在config文件中定义
        如果augment为True，说明此时需要使用扩增的数据，直接将其拼接在整个数据后面
        """
        cooler, valve, pump, hydraulic = self.information
        att_iter = product(cooler, valve, pump, hydraulic)
        att_list = []
        for item in att_iter:
            att_list.append(list(item))
        att_list = attribute_standard(att_list, self.information)

        indices = self.search_according_ways_num()
        self.data = self.data[indices]
        self.attribute = self.attribute[indices]

        train_indices = []
        # 提取出每一类的第一个样本的索引值
        # TODO 提取方法有待改进，两个循环的暴力搜索的时间复杂度有点高
        # TODO 这样子切割使得训练集与测试集完全固定，被筛选掉的样本永远也不会参与计算，可以改成每次创建dataset都是不同的
        for item in att_list:  # 遍历每一个类别
            num = 0  # 用来计数提取的属性个数
            self.classes.append(list(item))
            for idx, att in enumerate(self.attribute):  # 遍历每一个属性
                # att = tuple(att)
                # item = tuple(item)
                if num == self.shots_num:  # 如果已经提取完对应的样本数量就退出循环
                    break
                if np.all(item == att):  # 提取完第一个样本后
                    train_indices.append(idx)
                    num += 1

        # 拷贝，防止由于test的改变而导致原数据改变
        data = np.copy(self.data)
        attribute = np.copy(self.attribute)
        # self.test_data = data[test_indices]
        # self.test_attribute = attribute[test_indices]
        self.train_data = data[train_indices]
        self.train_attribute = attribute[train_indices]
        self.test_data = np.delete(data, train_indices, axis=0)
        self.test_attribute = np.delete(attribute, train_indices, axis=0)

        # 是否使用扩增的数据
        if self.augment and self.with_train:
            self.train_data = np.concatenate([self.train_data, self.augment_data], axis=0)
            self.train_attribute = np.concatenate([self.train_attribute, self.augment_attribute], axis=0)
        elif self.augment and not self.with_train:
            self.train_data = self.augment_data
            self.train_attribute = self.augment_attribute

        # np.ndarray -> torch.FloatTensor
        self.data = torch.FloatTensor(self.data)
        self.train_data = torch.FloatTensor(self.train_data)
        self.train_attribute = torch.FloatTensor(self.train_attribute)
        self.test_data = torch.FloatTensor(self.test_data)
        self.test_attribute = torch.FloatTensor(self.test_attribute)

    def search_according_ways_num(self):
        """
        根据ways_num挑选数据
        选择的规则是尽可能让更多的属性不相同
        :return: dict()
        """
        cooler, valve, pump, hydraulic = self.config.information
        # # 归一化属性信息
        cooler = cooler / np.max(cooler) * 100
        valve = valve / np.max(valve) * 100
        pump = pump / np.max(pump) * 100
        hydraulic = hydraulic / np.max(hydraulic) * 100

        # 根据ways_num生成类别
        ways = []
        cooler_len = len(cooler)
        valve_len = len(valve)
        pump_len = len(pump)
        hydraulic_len = len(hydraulic)
        for i in range(self.ways_num):
            attribute = [cooler[i % cooler_len],
                         valve[i % valve_len],
                         pump[i % pump_len],
                         hydraulic[i % hydraulic_len]]
            # 检查类别是否重复
            if attribute in ways:
                continue
            ways.append(attribute)
        self.ways = np.reshape(ways, (-1, 4))

        # 根据ways获取数据
        profile = self.attribute
        indices = []
        for way in ways:
            for idx, att in enumerate(profile):
                if np.all(way == att):
                    indices.append(idx)

        return indices

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
    with open("..\\configs\\config_0.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = EasyDict(config)

    dataset = FaultDataset(config, method='LDA')
    # print(dataset.__getitem__(1))
    # print(dataset.__len__())
    # print(dataset.test_data)
    # print(dataset.test_attribute)
    print(dataset.train_data.shape)
    print(dataset.train_attribute)
    # print(dataset.test_data == dataset.train_data)
    # print(dataset.classes)
    # indices = [1, 2, 3, 5]
    # test = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]
    # test = np.array(test)
    # dele = np.delete(test, indices, axis=0)
    # dele[0][0] = 0
    # print(test)
    # print(dele)
