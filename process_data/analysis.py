# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 21:32 2023/3/15
"""
1. 使用LDA对数据集进行降维
2. 使用PCA对数据集进行降维
"""
import numpy as np
import sklearn.preprocessing
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import dill
import pandas as pd
import yaml
from easydict import EasyDict
import math


def transform_attribute_to_label(attribute, information):
    """
    将多维的属性矩阵转换成一维的标签矩阵
    :param attribute: 属性矩阵 [batch_size, attribute_dim]，np.ndarray
    :param information: 每个属性具体的取值
    :return: 一维的标签矩阵
    """
    # 根据information中每个属性的取值顺序，重新赋值attribute
    attribute_copy = np.copy(attribute)
    attribute_copy = pd.DataFrame(attribute_copy, index=None, columns=None)
    for item_idx, item in enumerate(information):
        for idx, value in enumerate(item):
            attribute_copy.loc[attribute_copy[item_idx] == value, item_idx] = idx
    attribute_copy = torch.FloatTensor(attribute_copy.values)

    # 根据属性取值的范围计算系数（类似于进制转换）
    # 每一位都是任意进制，例如第一位是满2进1，到了第二位就是满3进1
    # 然后将这个任意进制四位数转换成十进制
    information_len = [len(information[0]), len(information[1]), len(information[2]), len(information[3])]
    information_len.reverse()
    information_len = torch.FloatTensor(information_len)
    information_cumprod = torch.cumprod(information_len, dim=0)
    information_prod_reverse = information_cumprod.__reversed__()
    information_coefficient = torch.ones_like(information_prod_reverse)
    information_coefficient[:-1] = information_prod_reverse[1:]

    # 矩阵乘法，计算标签值
    label_mat = torch.matmul(attribute_copy, information_coefficient)
    return label_mat


def lda(data, label, dim_out=64):
    """
    使用线性判别分析，对data进行降维
    :param data: 原始数据  [batch_size, dim]
    :param label: 标签  [batch_size]
    :param dim_out: 降维后的维度，需要注意的是，该维度最大为 标签中不同种类个数-1
    :return: 降维后的数据
    """
    # 检查dim_out是否合法
    class_num = torch.unique(label).shape[0]
    if dim_out >= class_num:
        raise ValueError(f"dim_out can not greater than class_num-1, class_num:{class_num}")

    # 进行lda降维
    lda_model = LinearDiscriminantAnalysis(n_components=dim_out)
    lda_model.fit(data, label)
    # print(lda_model._n_features_out)
    data_after_lda = lda_model.transform(data)
    # print(lda_model._max_components)
    # print(data_after_lda.shape)
    return data_after_lda


def pca(data, dim_out=64):
    """
    使用主成分分析，对数据进行降维
    :param data: 原始数据  [batch_size, dim]
    :param dim_out: 降维后的维度
    :return: 降维后的数据
    """
    pca_model = PCA(n_components=dim_out)
    pca_model.fit(data)
    data_after_pca = pca_model.fit_transform(data)
    return data_after_pca


def data_standard(data: np.ndarray, eps=False):
    """
    对数据进行标准化处理
    :param data:
    :param eps:
    :return:
    """
    std = np.std(data, axis=0)
    mean = np.mean(data, axis=0)
    standard_data = np.divide((data - mean), std, where=std != 0)
    return standard_data


def attribute_standard(attribute, information):
    """
    将属性归一化到[0, 100]之间
    :param attribute:
    :param information:
    :return:
    """
    max_list = []
    for item in information:
        max_list.append(np.max(item))
    max_list = np.array(max_list)
    standard_attribute = (attribute / max_list) * 100
    return standard_attribute


def information_standard(information):
    """
    由于属性进行了归一化，所以这里也需要将information进行归一化，以便于匹配属性的变化
    :param information:
    :return:
    """
    standard_information = []
    for item in information:
        max_value = np.max(item)
        standard_information.append(list((item / max_value) * 100))
    return standard_information


def dim_decay(data_dict, information, dim_out=64, method=None, standard=True):
    """
    对数据进行降维
    :param data_dict: 包含原始数据的字典
    :param information: 数据类别信息
    :param dim_out: 输出的数据维度
    :param method: 用什么方法进行降维，可选"LDA"和"PCA"，当为None时，不对数据进行处理
    :param standard: 是否对数据使用标准化
    :return: 包含降维后的数据的字典
    """
    if method is None:
        print("There is no method, you should set a method for dim_decay!")
        return data_dict

    attribute = data_dict["attribute"]
    source_data = data_dict["data"]
    data_dim_decay = None

    if method == 'LDA':
        label = transform_attribute_to_label(attribute, information)
        # print(label.shape)
        data_dim_decay = lda(source_data, label, dim_out)
        # print(data_dim_decay.shape)
    elif method == 'PCA':
        data_dim_decay = pca(source_data, dim_out)

    if standard:
        # data_dim_decay = data_standard(data_dim_decay)
        # standardscaler = StandardScaler()
        scaler = MinMaxScaler()
        data_dim_decay = scaler.fit_transform(data_dim_decay)
    attribute = attribute_standard(attribute, information)

    data_dict_af_dim_decay = {
        "data": data_dim_decay,
        "attribute": attribute
    }
    return data_dict_af_dim_decay


if __name__ == '__main__':
    with open("..\\configs\\config_0.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = EasyDict(config)

    save_path = config.save_path
    save_lda_path = config.save_lda_path
    save_pca_path = config.save_pca_path
    save_standard_pca_path = config.save_standard_pca_path
    # with open(save_path, 'rb') as f:
    #     data = dill.load(f)
    # attribute = data['attribute']
    information = [[3, 20, 100], [100, 90, 80, 73], [0, 1, 2], [130, 115, 100, 90]]
    test_attribute = [[3, 100, 1, 130]]
    # information = information_standard(information)
    # with open(config.save_lda_path, 'rb') as f:
    #     data = dill.load(f)
    # label = transform_attribute_to_label(data['attribute'], information)
    label = transform_attribute_to_label(test_attribute, information)
    print(label)
    # attribute_standard(attribute, information)
    # label = transform_attribute_to_label(attribute, information)
    # source_data = data['data']
    # data_af_lda = lda(source_data, label)
    # data_dict_af_lda = dict()
    # data_dict_af_lda['data'] = data_af_lda
    # data_dict_af_lda['attribute'] = attribute
    # print(data_af_lda.shape)
    # with open(save_lda_path, 'wb') as f_lda:
    #     dill.dump(data_dict_af_lda, f_lda)
    # from collections import Counter
    # count = Counter(label.numpy())
    # print(count)
    # print(max(count.values()))
    # data_after_dim_decay = dim_decay(data, information, method='PCA')
    # with open(save_standard_pca_path, 'wb') as f_pca:
    #     dill.dump(data_after_dim_decay, f_pca)
    # with open(save_pca_path, 'rb') as r_pca:
    #     data_dict = dill.load(r_pca)
    # print(data_dict['data'].shape)

    # test = [[1, 2, 3, 4, 5],
    #         [2, 3, 4, 5, 6],
    #         [3, 4, 5, 6, 7]]
    # test = np.array(test)
    # mean = np.mean(test, axis=0)
    # std = np.std(test, axis=0)
    # print(mean)
    # print(std)
    # s_test = (test - mean) / std
    # print(test)
    # print(s_test)
    # a = np.array([1])
    # b = np.array([0])
    # print(a/b)

