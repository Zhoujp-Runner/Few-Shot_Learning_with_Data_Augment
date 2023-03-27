# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 19:46 2023/3/19
"""
1.KNN算法
"""
import os.path

import numpy as np
import torch
import pandas as pd
from collections import Counter
from process_data.dataset import FaultDataset
import yaml
from easydict import EasyDict
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import dill

from process_data.analysis import transform_attribute_to_label


with open("..\\configs\\config_0.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

config = EasyDict(config)


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
    def __init__(self):
        super(MLPClassification, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 144)
        )

    def forward(self, x):
        return self.layers(x)


def train(data_set):
    epochs = 100

    dataloader = DataLoader(data_set, batch_size=128, shuffle=True)

    device = 'cpu'
    model = MLPClassification()
    model = model.to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        total_loss = 0
        pb_dataloader = tqdm(dataloader, desc=f"Epoch{epoch}: ")
        for data, attribute in pb_dataloader:
            data = data.to(device)
            attribute = attribute.to(device)

            out = model(data)
            loss = loss_func(out, attribute)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pb_dataloader.set_postfix_str(f"Loss = {loss.item()}")
            total_loss += loss.item()

        loss_for_epoch = total_loss / len(dataloader)
        # 保存模型
        if (epoch + 1) % 10 == 0:
            save_name = f'pca_5_shots_augment_ConcatLinear_epoch{epoch}_loss{loss_for_epoch}.pkl'
            save_root = '..\\experiments\\classifications\\MLP'
            save_path = os.path.join(save_root, save_name)
            model_dict = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            with open(save_path, 'wb') as f:
                dill.dump(model_dict, f)


def test(data_set):
    data = data_set.test_data  # [batch_size, 64]
    labels = data_set.test_attribute  # [batch_size, 144]
    model = MLPClassification()
    load_path = "..\\experiments\\classifications\\MLP\\pca_5_shots_augment_ConcatLinear_epoch99_loss1.231703862923534.pkl"
    with open(load_path, 'rb') as f:
        model_dict = dill.load(f)
    model.load_state_dict(model_dict["state_dict"])
    out = model(data)  # [batch_size, 144]
    values, indices = torch.max(out, dim=1, out=None)
    accuracy = torch.sum(indices == labels) / len(dataset.test_attribute)
    print(accuracy)


if __name__ == '__main__':
    da = np.array([[4, 4, 4], [3, 3, 3], [4, 4, 4], [4, 4, 4]])
    x_in = np.array([[2, 2, 2], [1, 1, 1]])
    la = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [3, 3, 3, 3]])
    dataset = FaultDataset(config, method='PCA', augment=True)
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
    # train(dataset)
    # MLP test
    label_1d_test = transform_attribute_to_label(dataset.test_attribute, config.information)
    dataset.test_attribute = torch.FloatTensor(label_1d_test)
    test(dataset)

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
