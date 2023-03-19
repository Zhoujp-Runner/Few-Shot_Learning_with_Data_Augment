# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 19:46 2023/3/19
"""
1.KNN算法
"""
import numpy as np
import pandas as pd
from collections import Counter


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
            # TODO 有待优化，count返回出来的结果可能shape不一定
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
                max_label.append(key)

        return max_label


if __name__ == '__main__':
    da = np.array([[4, 4, 4], [3, 3, 3], [4, 4, 4], [4, 4, 4]])
    x_in = np.array([[2, 2, 2], [1, 1, 1]])
    la = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [3, 3, 3, 3]])
    knn = KNN(da, la, k=4)
    print(knn.classify(x_in))
    # x_in = x_in[:, None, :]
    # dif = da - x_in
    # dis = np.sum(dif, axis=2, keepdims=True)
    # pd_dis = list()
    # la_pd = pd.DataFrame(la, index=None, columns=None)
    # dic = dict()
    # dic[(1, 2 , 3)] = 1
    # print(dic)
    # for i in dis:
        # pd_i = np.concatenate([i, la], axis=1)
        # pd_i = pd.DataFrame(pd_i, index=None, columns=None)
        # print(pd_i)

        # pd_i.sort_values(by=0, axis=1)
        # pdi_sort = pd_i[np.argsort(pd_i[:, 0])]
        # vote = pdi_sort[:2, 1:]
        # label = KNN.count(vote)
        # print(vote)
        # print(label)
