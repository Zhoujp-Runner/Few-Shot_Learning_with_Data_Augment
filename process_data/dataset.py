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


save_path = "..\\processed_data\\data_dict.pkl"
save_lda_path = "..\\processed_data\\data_after_lda.pkl"


class FaultDataset(Dataset):
    def __init__(self,
                 config,
                 mode='train',
                 with_test=False):
        """
        :param config: config文件
        :param mode: 'train' or 'test'
        :param with_test: 是否要将测试集包括进生成模型的训练集中
        """
        super(FaultDataset, self).__init__()

        self.data_path = config.save_lda_path
        if not os.path.exists(self.data_path):
            raise OSError(f"There is not a existed path. Error path: {self.data_path}")
        with open(self.data_path, 'rb') as f:
            self.source_data = dill.load(f)

        self.data = self.source_data["data"]
        self.attribute = self.source_data["attribute"]
        self.information = config.information

        self.mode = mode
        self.with_test = with_test
        self.train_data = None
        self.train_attribute = None
        self.test_data = None
        self.test_attribute = None
        self.classes = []

        self.divide_data()

    def divide_data(self):
        """
        将source_data中的数据分成train, test两部分
        将某一个类别的第一个样本划分为测试集
        即测试集中每个类别只有一个样本
        """
        cooler, valve, pump, hydraulic = self.information
        att_iter = product(cooler, valve, pump, hydraulic)

        indices = []
        # 提取出每一类的第一个样本的索引值
        # TODO 提取方法有待改进，两个循环的暴力搜索的时间复杂度有点高
        for item in att_iter:
            self.classes.append(list(item))
            for idx, att in enumerate(self.attribute):
                att = tuple(att)
                if item == att:
                    indices.append(idx)
                    break

        # 拷贝，防止由于test的改变而导致原数据改变
        data = np.copy(self.data)
        attribute = np.copy(self.attribute)
        self.test_data = data[indices]
        self.test_attribute = attribute[indices]
        self.train_data = np.delete(data, indices, axis=0)
        self.train_attribute = np.delete(attribute, indices, axis=0)

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
    dataset = FaultDataset(config, mode='test')
    # print(dataset.__getitem__(1))
    # print(dataset.__len__())
    # print(dataset.test_data)
    # print(dataset.test_attribute)
    print(dataset.classes)
    # indices = [1, 2, 3, 5]
    # test = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]
    # test = np.array(test)
    # dele = np.delete(test, indices, axis=0)
    # dele[0][0] = 0
    # print(test)
    # print(dele)
