# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 21:32 2023/3/15
"""
1. 使用LDA对数据集进行降维
2. 使用PCA对数据集进行降维
"""
import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
import dill
import pandas as pd
import yaml
from easydict import EasyDict


with open("..\\configs\\config_0.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

config = EasyDict(config)


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
    data_after_lda = lda_model.transform(data)
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


def dim_decay(data_dict, information, method=None):
    """
    对数据进行降维
    :param data_dict: 包含原始数据的字典
    :param information: 数据类别信息
    :param method: 用什么方法进行降维，可选"LDA"和"PCA"，当为None时，不对数据进行处理
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
        data_dim_decay = lda(source_data, label)
    elif method == 'PCA':
        data_dim_decay = pca(source_data)

    data_dict_af_dim_decay = {
        "data": data_dim_decay,
        "attribute": attribute
    }
    return data_dict_af_dim_decay


if __name__ == '__main__':
    save_path = config.save_path
    save_lda_path = config.save_lda_path
    save_pca_path = config.save_pca_path
    with open(save_path, 'rb') as f:
        data = dill.load(f)
    attribute = data['attribute']
    information = [[3, 20, 100], [100, 90, 80, 73], [0, 1, 2], [130, 115, 100, 90]]
    label = transform_attribute_to_label(attribute, information)
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
    # with open(save_pca_path, 'wb') as f_pca:
    #     dill.dump(data_after_dim_decay, f_pca)
    with open(save_pca_path, 'rb') as r_pca:
        data_dict = dill.load(r_pca)
    print(data_dict['data'].shape)
