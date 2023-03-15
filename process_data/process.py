# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 17:29 2023/3/14
"""
1. 读取数据
2. 降维
3. 保存数据
"""
import os

import numpy as np
import pandas as pd

import dill
import torch

dataset_root_path = "..\\dataset"
text_list = ["CE", "CP", "profile"]
save_path = "..\\processed_data\\data_dict.pkl"


def read_from_txt(text_list, dataset_root_path):
    """
    从txt文件中读取数据，存入字典
    :param text_list: 需要读取的文件名列表
    :param dataset_root_path: txt所在文件的根目录
    :return: 以文件名为键，数据为值的字典
    """
    if not isinstance(text_list, list):
        raise TypeError("text_list must be a list!")
    data_origin = dict()
    for text_name in text_list:
        if not isinstance(text_name, str):
            raise TypeError("text_name must be str!")
        path = os.path.join(dataset_root_path, text_name + ".txt")
        data = pd.read_csv(path, sep='\t', header=None, index_col=None)
        data_origin[text_name] = data

    return data_origin


def concat_according_profile(data_origin: dict, text_list):
    if "profile" not in data_origin.keys():
        raise ValueError("profile.txt must be loaded, it is not in the dict!")

    # 根据profile中的值拼接数据
    # 其实就是根据行数，对向量进行拼接
    value = []
    for key in text_list:
        if key == "profile":
            continue
        value.append(data_origin[key].values)
    data = np.concatenate(value, axis=1)
    # index = pd.MultiIndex.from_product([[i for i in range(len(data))], text_list])
    # pd_data = pd.DataFrame(data, index=index, columns=None)
    attribute = data_origin["profile"].values

    # np.ndarray -> torch.FloatTensor
    data = torch.FloatTensor(data)
    attribute = torch.FloatTensor(attribute)
    return data, attribute


if __name__ == '__main__':
    origin_data = read_from_txt(text_list, dataset_root_path)
    print(origin_data["CE"].values[0])
    print(origin_data["CP"].values[0])
    data, attr = concat_according_profile(origin_data, text_list)
    print(data.shape)
    print(data[0])
    save_dict = dict()
    # save_dict["data"] = data
    # save_dict["attribute"] = attr
    # with open(save_path, 'wb') as f:
    #     dill.dump(save_dict, f)
    # with open(save_path, 'rb') as f:
    #     save_dict = dill.load(f)
    # print(save_dict)
    # print(attr)
