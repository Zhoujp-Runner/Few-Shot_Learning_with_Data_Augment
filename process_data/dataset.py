# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 16:47 2023/3/15
"""
1. 加载处理后的数据
2. 划分数据集(先不划分数据集，先将扩散模型搭完后再划分数据)
3. 建立映射关系：__getitem__, __len__
"""
import math
import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import dill
import yaml
from easydict import EasyDict
from itertools import product
from sklearn.preprocessing import MinMaxScaler

from process_data.analysis import attribute_standard, information_standard, transform_attribute_to_label


class FaultDataset(Dataset):
    def __init__(self,
                 config,
                 mode='train',
                 method='LDA',
                 ways=None,
                 augment=False,
                 with_train=True,
                 with_test=False,
                 use_random_combination=False):
        """
        :param config: 配置参数
        :param mode: 'train' or 'test'
        :param method: 'LDA' or 'PCA' or 'Standard PCA' or \
                        'Split Standard PCA' or 'Split Standard Dim3 PCA' \
                        or 'LDA Standard' or 'Split LDA Standard' or 'Split LDA Standard Dim3'
        :param ways: 默认为None，如果不为None，则指定了ways，不需要自动生成ways
        :param augment: 表示是否使用扩增数据集
        :param with_train: 扩增数据集是否包含原训练数据
        :param with_test: 是否要将测试集包括进生成模型的训练集中
        :param use_random_combination: 是否对所有的随机生成的类别组合分别进行训练
        """
        super(FaultDataset, self).__init__()
        self.dataset_type = 'Hydraulic'

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
        self.attribute = self.source_data["attribute"]  # 已经进行了归一化，区间为[0, 100]
        self.information = config.information

        self.config = config
        self.mode = mode
        self.augment = augment
        self.method = method
        self.with_train = with_train
        self.with_test = with_test
        self.use_random_combination = use_random_combination
        self.ways = ways
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

        if self.ways is None:
            # 根据ways_num获取组合
            if self.use_random_combination:
                indices = self.search_random_combination_ways()
            else:
                indices = self.search_according_ways_num()
        else:
            # 根据ways获取数据
            profile = self.attribute
            indices = []
            for way in self.ways:
                for idx, att in enumerate(profile):
                    if np.all(way == att):
                        indices.append(idx)
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

    def search_random_combination_ways(self):
        """
        根据ways_num获取随机的类别组合
        """
        # 得到所有的属性
        attribute_list = np.unique(self.attribute, axis=0)
        attribute_num = np.arange(len(attribute_list))

        random_indices = np.random.choice(attribute_num, self.ways_num, replace=False)
        random_combination = attribute_list[random_indices]
        self.ways = random_combination

        # 根据ways获取数据
        profile = self.attribute
        indices = []
        for way in random_combination:
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


class TEPDataset(Dataset):
    def __init__(self,
                 config,
                 mode='train',
                 shots_num=None,
                 ways_num=None,
                 ways=None,
                 augment=False):
        super(TEPDataset, self).__init__()
        self.dataset_type = 'TEP'
        self.config = config
        self.mode = mode
        self.augment = augment

        self.source_path = config.tep_train_lda_standard_path
        if not os.path.exists(self.source_path):
            raise ValueError("There is not a such file of tep dataset!")
        with open(self.source_path, 'rb') as f:
            self.source_data = dill.load(f)

        if shots_num is None:
            self.shots_num = config.shots_num
        else:
            self.shots_num = shots_num
        if ways_num is None:
            self.ways_num = config.ways_num
        else:
            self.ways_num = ways_num
        self.expand_num = 9

        self.total_classes = np.arange(1, 22)  # TEP数据集总共有21种故障类型
        self.train_data = []
        self.test_data = []
        self.classification_data = []
        self.ways = None
        # self.get_data_randomly()
        self.get_data_randomly_for_specified_way()

        if self.mode == 'train':
            self.len = len(self.train_data)
        elif self.mode == 'test':
            self.len = len(self.test_data)

    def get_data_randomly(self):
        """
        根据ways_num和shots_num，随机地从数据集中选取数据
        :return:
        """
        # 随机选择ways
        self.ways = np.random.choice(self.total_classes, size=self.ways_num, replace=False)
        # 根据ways以及shots_num，随机选择shots
        index = np.arange(0, 480)  # 每一类中有480个样本
        for way in self.ways:
            data_of_way = []
            for item in self.source_data:
                if item[-1] == way:
                    data_of_way.append(item)
            # TODO 这里注意一下为什么concatenate和stack的效果会不一样
            data_of_way = np.stack(data_of_way, axis=0)
            # 随机选择shots，注意由于data不是一维的，所以需要用索引矩阵
            index_of_shots = np.random.choice(index, size=self.shots_num, replace=False)
            shots_of_way = data_of_way[index_of_shots]
            data_copy = np.copy(data_of_way)
            data_except_shots = np.delete(data_copy, index_of_shots, axis=0)
            self.train_data.append(shots_of_way)
            self.test_data.append(data_except_shots)

        self.train_data = np.concatenate(self.train_data, axis=0)  # 最终的数据集 shape:[ways_num * shots_num, 53]
        self.test_data = np.concatenate(self.test_data, axis=0)

        if self.augment:
            save_augment_root = self.config.augment_data_root
            save_augment_name = f"{self.config.shots_num}_{self.config.dataset_type}_{self.config.augment_num}.pkl"
            save_augment_path = os.path.join(save_augment_root, save_augment_name)
            with open(save_augment_path, 'rb') as f:
                self.train_data = dill.load(f)

        self.train_data = torch.FloatTensor(self.train_data)
        self.test_data = torch.FloatTensor(self.test_data)

    def get_data_randomly_for_specified_way(self):
        # 随机抽取类别
        self.ways = np.random.choice(self.total_classes, size=self.ways_num, replace=False)
        # 如果随机抽取的类别中没有6，就将第一个类别改为6
        if 1 not in self.ways:
            self.ways[0] = 1

        for way in self.ways:
            # way = 1
            index = np.arange(0, 480)  # 每一类中有480个样本
            data_of_way = []
            for item in self.source_data:
                if item[-1] == way:
                    data_of_way.append(item)
            # TODO 这里注意一下为什么concatenate和stack的效果会不一样
            data_of_way = np.stack(data_of_way, axis=0)
            # 如果类别是1，那么就取147, 328, 8, 144, 252[147, 8, 252][]
            # 类别6：129 148 321 340  37 141  79 [37, 148, 321]
            if way == 1:
                index_of_shots = np.array([328, 8, 252], dtype=np.int64)
            else:  # 否则就随机选取数据
                index_of_shots = np.random.choice(index, size=self.shots_num, replace=False)
            shots_of_way = data_of_way[index_of_shots]
            # 计算余弦相似度
            cosine, cosine_sum, weight = self.calculate_cosine_relationship(shots_of_way)
            # 根据权重扩张数据
            expanded_data_of_way = self.expand_data(shots_of_way, weight)
            if way == 1:
                print(cosine)
                print(cosine_sum)
                print(weight)
                print(shots_of_way)
                print(expanded_data_of_way)
            data_copy = np.copy(data_of_way)
            data_except_shots = np.delete(data_copy, index_of_shots, axis=0)
        # return shots_of_way, data_except_shots, index_of_shots
        #     self.train_data.append(shots_of_way)
            self.train_data.append(expanded_data_of_way)
            self.test_data.append(data_except_shots)
            # self.classification_data.append(shots_of_way)

        self.train_data = np.concatenate(self.train_data, axis=0)  # 最终的数据集 shape:[ways_num * shots_num, 16 + 1]
        self.test_data = np.concatenate(self.test_data, axis=0)
        # self.classification_data = np.concatenate(self.classification_data, axis=0)
        # print(self.train_data.shape)
        # print(self.test_data.shape)

        if self.augment:
            save_augment_root = self.config.augment_data_root
            save_augment_name = f"{self.config.shots_num}_{self.config.dataset_type}_{self.config.augment_num}.pkl"
            save_augment_path = os.path.join(save_augment_root, save_augment_name)
            with open(save_augment_path, 'rb') as f:
                self.train_data = dill.load(f)

        self.train_data = torch.FloatTensor(self.train_data)
        self.test_data = torch.FloatTensor(self.test_data)
        # self.classification_data = torch.FloatTensor(self.classification_data)

    @staticmethod
    def calculate_cosine_relationship(vectors):
        """
        计算每一个数据点与其他数据点之间的余弦相似度，其中与自己的相似度设置为0（实际上为1，但为了方便后面的计算，取为0）
        :param vectors: [shots_num, dim]
        :return: [shots_num, shots_num], [shots_num]  前者是某一个样本与其他样本的余弦相似度，后者是将某一个样本与所有样本余弦相似度求和
        """
        shots_num = vectors.shape[0]
        res = np.zeros((shots_num, shots_num))
        for idx, vector in enumerate(vectors):
            for another_idx, another in enumerate(vectors):
                if idx != another_idx:
                    # 向量的点乘
                    project = np.dot(vector, another)
                    # 两个向量模长的乘积
                    norm_prod = np.linalg.norm(vector) * np.linalg.norm(another)
                    # 余弦
                    res[idx, another_idx] = project / norm_prod
        res_sum = np.sum(res, axis=1)
        res_e_sum = np.exp(2 * res_sum)
        total_sum = np.sum(res_e_sum)
        res_weight = res_e_sum / total_sum
        return res, res_sum, res_weight

    def expand_data(self, vectors, weights):
        """
        根据prob中每个向量对应的权重大小，分别扩张向量的数量（只是简单的复制）
        :param vectors: [shots_num, dim]
        :param weights: [shots_num]
        :return: [shots_num + expand_num, dim]
        """
        res = []
        expand_num_sum = 0
        for idx, weight in enumerate(weights):
            if idx == self.shots_num - 1:
                expand_num_for_this_vector = self.expand_num - expand_num_sum
            else:
                expand_num_for_this_vector = round(self.expand_num * weight)
            expand_num_sum += expand_num_for_this_vector
            vector = vectors[idx]
            if expand_num_for_this_vector <= 0:
                continue
            res.append(vector)
            for _ in range(expand_num_for_this_vector - 1):
                noise = np.random.randn(*vector.shape) / 100
                res.append(vector.copy() + noise)
                # res.append(vector.copy())
        res = np.stack(res, axis=0)
        if res.shape[0] != self.expand_num:
            raise ValueError(f"expand num is wrong! {self.expand_num} is needed, but result is {res.shape}")
        return res

    def __getitem__(self, item):
        """返回数据和标签值"""
        if self.mode == 'train':
            sample_with_label = self.train_data[item]
        elif self.mode == 'test':
            sample_with_label = self.test_data[item]
        else:
            raise ValueError(f"No such mode:{self.mode} !")
        sample = sample_with_label[:-1]
        label = sample_with_label[-1].int()
        return sample, label

    def __len__(self):
         return self.len


class GuidedDataset(Dataset):
    def __init__(self,
                 config,
                 dataset_type):
        super(GuidedDataset, self).__init__()
        self.dataset_type = dataset_type
        self.config = config

        if dataset_type == 'TEP':
            self.source_path = config.tep_train_lda_standard_path
        elif dataset_type == 'Hydraulic':
            self.source_path = config.save_lda_standard_path
        else:
            raise ValueError("No such dataset type")

        if not os.path.exists(self.source_path):
            raise ValueError("There is not a such file of tep dataset!")
        with open(self.source_path, 'rb') as f:
            self.source_data = dill.load(f)

        if dataset_type == 'TEP':
            self.train_data = torch.FloatTensor(self.source_data)
        elif dataset_type == 'Hydraulic':
            self.data = torch.FloatTensor(self.source_data["data"])
            self.attribute = self.source_data["attribute"]
            information = information_standard(self.config.information)
            self.label_1d = transform_attribute_to_label(self.attribute, information)
            self.label_1d = self.label_1d[..., None]
            self.train_data = torch.cat([self.data, self.label_1d], dim=-1)

        self.len = len(self.train_data)

    def __getitem__(self, item):
        """返回数据和标签值"""
        sample_with_label = self.train_data[item]
        sample = sample_with_label[:-1]
        label = sample_with_label[-1].int()
        return sample, label

    def __len__(self):
        return self.len


if __name__ == '__main__':
    with open("..\\configs\\config_0.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = EasyDict(config)
    # # 液压数据集
    # dataset = FaultDataset(config, method='LDA Standard', use_random_combination=True)
    # dataset2 = FaultDataset(config, method='LDA', ways=dataset.ways, use_random_combination=True)
    # print(type(dataset.ways))
    # inf = information_standard(config.information)
    # label = transform_attribute_to_label(dataset.ways, inf)
    # print(label)
    # print(dataset2.ways)
    # print(dataset.train_data)
    # print(dataset2.train_data)
    # print(dataset.train_attribute)
    # print(dataset2.train_attribute)
    # print(dataset.__getitem__(1))
    # print(dataset.__len__())
    # print(dataset.test_data)
    # print(dataset.test_attribute)
    # print(dataset.train_data.shape)
    # print(dataset.train_attribute)
    # print(dataset.test_data == dataset.train_data)
    # print(dataset.classes)
    # indices = [1, 2, 3, 5]
    # test = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]]
    # test = np.array(test)
    # dele = np.delete(test, indices, axis=0)
    # dele[0][0] = 0
    # print(test)
    # print(dele)

    # TEP数据集
    tep_dataset = TEPDataset(config, mode='test')
    teet = [[1, 1],
            [2, 1],
            [0, 2]]
    teet = np.array(teet)
    _, __, wei = tep_dataset.calculate_cosine_relationship(teet)
    # print(_, __, wei)
    # print(tep_dataset.expand_data(teet, wei))
    print(tep_dataset.train_data.shape)
    print(tep_dataset.ways)
    # train_data, test_data, index = tep_dataset.get_data_randomly_for_specified_way()
    # print(index)
    # data = np.concatenate([train_data, test_data], axis=0)
    # way = torch.FloatTensor(tep_dataset.ways)
    # for w in way:
    #     w = w.unsqueeze(0)
    #     w = w.expand(10, 1)
    #     print(w)
    # x, y = tep_dataset.__getitem__(0)
    # print(x)
    # print(y.type())
    # print(tep_dataset.__len__())
    # test_data = tep_dataset.test_data
    # # data, labels = torch.split(test_data, [52, 1], dim=-1)
    # # print(data.shape)
    # # print(labels)
    # # labels = labels.view(-1)
    # # print(labels)
    # # x = torch.IntTensor([1, 2, 3, 2, 1])
    # # y = torch.FloatTensor([1, 2, 5, 2, 1])
    # # print(torch.sum(x == y))
    # import umap
    # from sklearn.manifold import TSNE, MDS
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(5, 6)
    # for i in range(1, 31):
    #     u_map = umap.UMAP(n_neighbors=i+1, unique=True)
    #     data_viz = u_map.fit_transform(data)
    #     # tsne = TSNE(n_components=2, perplexity=i+1)
    #     # data_viz = tsne.fit_transform(data)
    #     x0 = data_viz[:3, 0]
    #     y0 = data_viz[:3, 1]
    #     x1 = data_viz[3:, 0]
    #     y1 = data_viz[3:, 1]
    #     index = i - 1
    #     row = index // 6
    #     col = index % 6
    #     axes[row][col].scatter(x1, y1, c='green')
    #     axes[row][col].scatter(x0, y0, c='red')
    #     axes[row][col].annotate("328", (x0[0], y0[0]))
    #     axes[row][col].annotate("8", (x0[1], y0[1]))
    #     axes[row][col].annotate("252", (x0[2], y0[2]))
    # plt.show()

    # # 引导分类器数据集
    # guided_dataset = GuidedDataset(config, 'TEP')
    # print(guided_dataset.train_data.shape)
    # print(guided_dataset.len)
    # print(guided_dataset.__getitem__(999))
    # print(guided_dataset.train_data.shape)
    # print(guided_dataset.label_1d[0])
