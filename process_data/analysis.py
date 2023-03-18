# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 21:32 2023/3/15
"""
1. 使用LDA对数据集进行降维
"""
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import dill
import pandas as pd


save_path = "..\\processed_data\\data_dict.pkl"
save_lda_path = "..\\processed_data\\data_after_lda.pkl"


def transform_attribute_to_label(attribute, information):
    """
    将多维的属性矩阵转换成一维的标签矩阵
    :param attribute: 属性矩阵 [batch_size, attribute_dim]，可以是tensor，也可以是ndarray
    :param information: 每个属性具体的取值
    :return: 一维的标签矩阵
    """
    # 根据information中每个属性的取值顺序，重新赋值attribute
    attribute = pd.DataFrame(attribute, index=None, columns=None)
    for item_idx, item in enumerate(information):
        for idx, value in enumerate(item):
            attribute.loc[attribute[item_idx] == value, item_idx] = idx
    attribute = torch.FloatTensor(attribute.values)

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
    label_mat = torch.matmul(attribute, information_coefficient)
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


if __name__ == '__main__':
    with open(save_path, 'rb') as f:
        data = dill.load(f)
    attribute = data['attribute']
    information = [[3, 20, 100], [100, 90, 80, 73], [0, 1, 2], [130, 115, 100, 90]]
    label = transform_attribute_to_label(attribute, information)
    source_data = data['data']
    data_af_lda = lda(source_data, label)
    data_dict_af_lda = dict()
    data_dict_af_lda['data'] = data_af_lda
    data_dict_af_lda['attribute'] = attribute
    print(data_af_lda.shape)
    with open(save_lda_path, 'wb') as f_lda:
        dill.dump(data_dict_af_lda, f_lda)
