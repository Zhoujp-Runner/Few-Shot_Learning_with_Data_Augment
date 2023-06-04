# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 19:46 2023/3/19
"""
1.KNN算法
"""
import os.path
import sys

import numpy as np
import torch
import pandas as pd
from collections import Counter
from process_data.dataset import FaultDataset, TEPDataset
import yaml
from easydict import EasyDict
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import dill
import logging

from process_data.analysis import transform_attribute_to_label, information_standard


def makedir(path):
    """
    如果路径不存在，创建文件夹
    :param path: 文件夹路径
    """
    if not os.path.exists(path):
        os.makedirs(path)


class KNN(object):
    """KNN分类算法"""
    def __init__(self,
                 dataset: np.ndarray,
                 labels: np.ndarray,
                 k=20):
        """
        :param dataset: 数据集 [data_size, dim]
        :param labels: 对应的标签 [data_size, 4]
        :param k: knn算法的参与投票的样本数量
        """
        self.dataset = dataset
        self.data_size = dataset.shape[0]
        self.labels = labels
        self.k = k

    def classify(self, x: np.ndarray):
        """
        :param x: 需要被分类的样本 [batch_size, dim]
        :return: classified result [batch_size, dim]
        """
        batch_size = x.shape[0]
        x = x[:, None, :]
        diff_mat = self.dataset - x
        distance = np.sqrt(np.sum(np.square(diff_mat), axis=2, keepdims=True))
        result = list()
        for item in distance:
            item = np.concatenate([item, self.labels], axis=1)
            # pd_item = pd.DataFrame(item, index=None, columns=None)
            # pd_item.sort_values(axis=0)
            # TODO 有待优化，如果第k个和第k+1个数值相同，那么k+1就会被舍弃
            item_sort = item[np.argsort(item[:, 0])]
            voter = item_sort[:self.k + 1, 1:]
            # TODO 有待优化，count当有两个票数相同的标签时只返回一个
            label = self.count(voter)
            result.append(label)
        result = np.reshape(result, (batch_size, 4))
        return result

    @staticmethod
    def count(voter):
        """
        计算label出现的次数
        :param voter: 参与投票的数组 [k, 4]
        :return: 返回得票最多的label
        """
        # print(voter)
        count_dict = dict()
        for item in voter:
            # print(item)
            item = tuple(item)
            # 如果当前的标签值未被统计过
            if item not in count_dict.keys():
                count_dict[item] = 1
                continue
            # 如果当前的标签值已被统计过
            count_dict[item] += 1
        # print(count_dict)
        max_label = []
        for key, value in count_dict.items():
            if value == max(count_dict.values()):
                max_label.append(list(key))
                break

        return max_label

    def count_accuracy(self,
                       prediction: np.ndarray,
                       ground_truth: np.ndarray):
        mask = prediction == ground_truth
        result = [1 if np.all(item) else 0 for item in mask]
        accuracy = np.sum(result) / len(ground_truth)
        return accuracy

    def run(self,
            test_data: np.ndarray,
            test_label: np.ndarray):
        predition = self.classify(test_data)
        accuracy = self.count_accuracy(predition, test_label)
        return accuracy


class MLPClassification(nn.Module):
    def __init__(self, dim_in, dim3=False, dataset='Hydraulic'):
        super(MLPClassification, self).__init__()
        if dim3:
            self.layers = nn.Sequential(
                nn.Linear(dim_in, dim_in//2),
                nn.ReLU(),
                nn.Linear(dim_in//2, dim_in//4),
                nn.ReLU(),
                nn.Linear(dim_in//4, 144)
            )
        elif dataset == 'Hydraulic':
            self.layers = nn.Sequential(
                nn.Linear(dim_in, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 144)
            )
        elif dataset == 'TEP':
            self.layers = nn.Sequential(
                nn.Linear(dim_in, 128),
                nn.ReLU(),
                nn.Linear(128, 21)
            )
        self.type = 'FullConnected'

    def forward(self, x):
        return self.layers(x)


class TrainClassification(object):

    def __init__(self,
                 config,
                 ways):
        super(TrainClassification, self).__init__()
        self.config = config
        self.ways=ways
        self.use_augment = [False, True]
        self._set_log()
        self.is_transform = False

    def _set_log(self):
        """设置log文件"""
        self.logger = logging.getLogger("ClassificationLog")
        self.logger.setLevel(logging.DEBUG)

        # 清空该log的句柄
        for handle in self.logger.handlers:
            self.logger.removeHandler(handle)

        # self.filehandle = logging.FileHandler(self.config.save_log_path)
        file_root = self.config.classification_root
        if self.config.dataset_type == 'Hydraulic':
            file_name = f"classification_{self.config.shots_num}_{self.config.method}.log"
        elif self.config.dataset_type == 'TEP':
            file_name = f"classification_{self.config.shots_num}_{self.config.dataset_type}.log"
        else:
            raise ValueError('Please use the right dataset!')
        file_path = os.path.join(file_root, file_name)
        self.filehandle = logging.FileHandler(file_path)
        self.filehandle.setLevel(logging.DEBUG)

        fmt = "%(message)s"
        self.formatter = logging.Formatter(fmt)

        self.filehandle.setFormatter(self.formatter)
        self.logger.addHandler(self.filehandle)

    def train_loop(self, time=0, data_set=None):
        if data_set is not None:
            test_data = data_set.test_data
        result = []
        with_train = False
        self.logger.info(f"time: {time}")
        for use_augment in self.use_augment:
            if data_set is None:
                dataset = FaultDataset(self.config,
                                       method=self.config.method,
                                       ways=self.ways,
                                       augment=use_augment,
                                       with_train=with_train)
                self.logger.info(f"ways: {dataset.ways}")
                self.logger.info(f"with_train is {with_train}")
                max_epoch, max_accuracy = self.train(dataset)
                result.append([max_epoch, max_accuracy])
            elif data_set is not None and use_augment is False:
                self.logger.info(f"ways: {data_set.ways}")
                max_epoch, max_accuracy = self.train(data_set)
                result.append([max_epoch, max_accuracy])
            elif data_set is not None and use_augment is True:
                dataset = TEPDataset(self.config,
                                     augment=use_augment)
                dataset.test_data = test_data
                max_epoch, max_accuracy = self.train(dataset)
                result.append([max_epoch, max_accuracy])
        return result

    def train(self, data_set):
        self.logger.info(f"=============================Augment:{data_set.augment}================================")
        epochs = 100
        # 初始化标签转换的判断变量
        self.is_transform = False
        print(data_set.train_data.shape)
        if data_set.dataset_type == 'Hydraulic':
            # 将四维的属性向量转化为144维的类别向量
            information = information_standard(self.config.information)
            label_1d = transform_attribute_to_label(data_set.train_attribute, information)
            print(label_1d)
            label_list = []
            for idx in label_1d:
                label = torch.zeros((1, 144))
                idx = int(idx.item())
                label[0][idx] = 1
                label_list.append(label)
            label = torch.cat(label_list, dim=0)
            data_set.train_attribute = torch.FloatTensor(label)

        if (self.config.method == 'Split Standard Dim3 PCA' or self.config.method == 'Split LDA Standard Dim3') \
                and data_set.dataset_type == 'Hydraulic':
            train_set_size = data_set.train_data.shape[0]
            test_set_size = data_set.test_data.shape[0]
            data_set.train_data = data_set.train_data.contiguous().view(train_set_size, -1)
            data_set.test_data = data_set.test_data.contiguous().view(test_set_size, -1)

        dim_in = data_set.train_data.shape[-1]
        if data_set.dataset_type == 'TEP':
            dim_in = data_set.train_data.shape[-1] - 1  # 因为最后一维是标签

        dataloader = DataLoader(data_set, batch_size=128, shuffle=True)

        device = 'cpu'
        model = MLPClassification(dim_in, dataset=data_set.dataset_type)
        model = model.to(device)

        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

        max_accuracy = 0
        max_epoch = 0
        for epoch in range(epochs):
            total_loss = 0
            pb_dataloader = tqdm(dataloader, desc=f"Epoch{epoch}: ")
            for data, attribute in pb_dataloader:
                batch_size = data.shape[0]
                data = data.to(device)
                if data_set.dataset_type == 'TEP':
                    one_hot = torch.zeros(batch_size, 21)
                    attribute = attribute.view(-1).long()  # [batch_size, 1] -> [batch_size, ]
                    for index, y in enumerate(attribute):
                        one_hot[index][y-1] = 1
                    attribute = one_hot
                    attribute = attribute.to(device)
                elif data_set.dataset_type == 'Hydraulic':
                    attribute = attribute.to(device)

                out = model(data)
                print(out)
                print(attribute)
                loss = loss_func(out, attribute)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pb_dataloader.set_postfix_str(f"Loss = {loss.item()}")
                total_loss += loss.item()

            loss_mean = total_loss / len(dataloader)
            self.logger.info(f"Epoch{epoch}: train_loss = {loss_mean}")

            test_accuracy = self.test(data_set, dim_in, model=model)
            self.logger.info(f"Test Accuracy = {test_accuracy}")
            if max_accuracy < test_accuracy:
                max_accuracy = test_accuracy
                max_epoch = epoch

            # 保存模型
            if (epoch + 1) % 10 == 0:
                save_name = f'epoch{epoch}.pkl'
                save_root = self.config.classification_model_root
                save_dir = os.path.join(save_root, model.type)
                makedir(save_dir)
                save_path = os.path.join(save_dir, save_name)
                # model_dict = {
                #     "state_dict": model.state_dict(),
                #     "optimizer": optimizer.state_dict()
                # }
                model_dict = model.state_dict()
                with open(save_path, 'wb') as f:
                    dill.dump(model_dict, f)
        self.logger.info(f"max_epoch {max_epoch} : max_accuracy = {max_accuracy}")
        return max_epoch, max_accuracy

    def test(self, data_set, dim_in, model=None, load_path=None):
        with torch.no_grad():
            # 确保数据集不会进行两次标签转换
            if data_set.dataset_type == 'Hydraulic':
                if not self.is_transform:
                    information = information_standard(self.config.information)
                    label_1d_test = transform_attribute_to_label(data_set.test_attribute, information)
                    data_set.test_attribute = torch.FloatTensor(label_1d_test)
                    self.is_transform = True

                data = data_set.test_data  # [batch_size, 64]
                labels = data_set.test_attribute  # [batch_size, 144]

            elif data_set.dataset_type == 'TEP':
                data_with_labels = data_set.test_data  # [..., 53]
                data, labels = torch.split(data_with_labels, [16, 1], dim=-1)
                labels = labels - 1
                labels = labels.view(-1)

            if load_path is not None and model is None:
                model = MLPClassification(dim_in)
                with open(load_path, 'rb') as f:
                    model_dict = dill.load(f)
                model.load_state_dict(model_dict)
            elif load_path is None and model is None:
                raise ValueError("load path and model are both None!")

            out = model(data)  # [batch_size, 144]
            values, indices = torch.max(out, dim=1, out=None)
            # print(indices)
            # print(labels)
            # print(torch.sum(indices == labels))
            print(len(data_set.test_data))
            accuracy = torch.sum(indices == labels) / len(data_set.test_data)
        return accuracy.item()


if __name__ == '__main__':
    with open("..\\configs\\config_0.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = EasyDict(config)

    # da = np.array([[4, 4, 4], [3, 3, 3], [4, 4, 4], [4, 4, 4]])
    # x_in = np.array([[2, 2, 2], [1, 1, 1]])
    # la = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [3, 3, 3, 3]])
    # dataset = FaultDataset(config, method='Split Standard Dim3 PCA')

    # dim_in = dataset.train_data.shape[-1]
    # train_set_size = dataset.train_data.shape[0]
    # test_set_size = dataset.test_data.shape[0]
    # dataset.train_data = dataset.train_data.contiguous().view(train_set_size, -1)
    # dataset.test_data = dataset.test_data.contiguous().view(test_set_size, -1)
    # print(dataset.train_data.shape)
    # print(dataset.test_data.shape)
    # print(dataset.train_attribute.shape)
    # # KNN
    # train_data = dataset.train_data.numpy()
    # label = dataset.train_attribute.numpy()
    # knn = KNN(train_data, label, k=5)
    # test_data = dataset.test_data.numpy()
    # test_label = dataset.test_attribute.numpy()
    # print(knn.run(test_data, test_label))

    # # MLP train
    # label_1d = transform_attribute_to_label(dataset.train_attribute, config.information)
    # label_list = []
    # for idx in label_1d:
    #     label = torch.zeros((1, 144))
    #     idx = int(idx.item())
    #     label[0][idx] = 1
    #     label_list.append(label)
    # label = torch.cat(label_list, dim=0)
    # dataset.train_attribute = torch.FloatTensor(label)
    # train(dataset, dim_in)
    # # MLP test
    # label_1d_test = transform_attribute_to_label(dataset.test_attribute, config.information)
    # dataset.test_attribute = torch.FloatTensor(label_1d_test)
    # test(dataset, dim_in)

    # x_in = x_in[:, None, :]
    # dif = da - x_in
    # dis = np.sum(dif, axis=2, keepdims=True)
    # pd_dis = list()
    # la_pd = pd.DataFrame(la, index=None, columns=None)
    # dic = dict()
    # dic[(1, 2 , 3)] = 1
    # print(dic)
    # for i in dis:
    #     pd_i = np.concatenate([i, la], axis=1)
    #     pd_i = pd.DataFrame(pd_i, index=None, columns=None)
    #     print(pd_i)
    #
    #     pd_i.sort_values(by=0, axis=1)
    #     pdi_sort = pd_i[np.argsort(pd_i[:, 0])]
    #     vote = pdi_sort[:2, 1:]
    #     label = KNN.count(vote)
    #     print(vote)
    #     print(label)

    # x = torch.randn(32, 17, 32)
    # x = x.contiguous().view(32, -1)
    # model = MLPClassification(544, dim3=True)
    # print(model(x).shape)

    tep_dataset = TEPDataset(config)
    tep_dataloader = DataLoader(tep_dataset, batch_size=10, shuffle=True)
    for x, y in tep_dataloader:
        shape = x.shape[0]
        o = torch.zeros(shape, 21)
        y = y.long()
        print(y)
        for index, y_ in enumerate(y):
            o[index][y_-1] = 1
        print(o)
