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

import yaml
from easydict import EasyDict
from itertools import product

from analysis import dim_decay, attribute_standard

# dataset_root_path = "..\\dataset"
# text_list = ["CE", "CP", "profile"]
# save_path = "..\\processed_data\\data_dict.pkl"


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


def search_according_shots_num(data_origin, ways_num, information):
    """
    根据ways_num挑选数据
    选择的规则是尽可能让更多的属性不相同
    :param data_origin: 原始数据
    :param ways_num: 一共所需要训练的类别数量
    :param information: 属性的取值信息
    :return: dict()
    """
    cooler, valve, pump, hydraulic = information
    # # 归一化属性信息
    # cooler = cooler / np.max(cooler) * 100
    # valve = valve / np.max(valve) * 100
    # pump = pump / np.max(pump) * 100
    # hydraulic = hydraulic / np.max(hydraulic) * 100

    # 根据ways_num生成类别
    ways = []
    cooler_len = len(cooler)
    valve_len = len(valve)
    pump_len = len(pump)
    hydraulic_len = len(hydraulic)
    for i in range(ways_num):
        attribute = [cooler[i % cooler_len],
                     valve[i % valve_len],
                     pump[i % pump_len],
                     hydraulic[i % hydraulic_len]]
        # 检查类别是否重复
        if attribute in ways:
            continue
        ways.append(attribute)
    ways = np.reshape(ways, (-1, 4))
    print(ways)

    # 根据ways获取数据
    profile = data_origin["profile"].values[:, :-1]
    indices = []
    result = dict()
    for way in ways:
        for idx, att in enumerate(profile):
            if np.all(way == att):
                indices.append(idx)
    for key in data_origin.keys():
        result[key] = data_origin[key].values[indices]

    return result


def concat_according_profile(data_origin: dict, text_list):
    """
    根据类别，沿着第1维度对数据进行拼接
    :param data_origin: 原始数据字典
    :param text_list: 文件列表
    :return: [N, dims]
    """
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

    # # np.ndarray -> torch.FloatTensor
    # data = torch.FloatTensor(data)
    # attribute = torch.FloatTensor(attribute[:, :-1])
    return data, attribute


def concat_2d(data_origin: dict, text_list):
    """
    根据类别，沿着第0维度对数据进行拼接
    :param data_origin: 原始数据字典
    :param text_list: 文件列表
    :return: [N, attribute_num, dim]
    """
    if "profile" not in data_origin.keys():
        raise ValueError("profile.txt must be loaded, it is not in the dict!")

    attribute = data_origin['profile'].values

    datas = []
    for i in range(len(attribute)):
        value = []
        for key in data_origin.keys():
            if key == "profile":
                continue
            value.append(data_origin[key].values)
        data = np.concatenate(value, axis=0)
        datas.append(data)
    # datas = np.concatenate(datas, axis=0)

    print(attribute)
    # print(datas.shape)


def process(config, split=False, dim3=False, standard=True):
    """
    对数据进行预处理
    :param config: 配置文件
    :param split: 是否按照属性分别对数据进行处理，最后进行合并，如果为False，则一开始对数据在第0维进行拼接，然后降维
    :param dim3: 只有一种情况该参数有效，即如果split为True，且dim3为True，那么数据会转换成三维的，即[N, attribute_num, 8]
    :param standard: 是否对数据进行归一化
    """
    # 加载参数
    text_list = config.text_list
    dataset_root_path = config.dataset_root_path
    information = config.information
    save_path = config.save_path
    save_lda_path = config.save_lda_path
    save_standard_pca_path = config.save_standard_pca_path
    save_standard_split_pca_path = config.save_standard_split_pca_path
    save_split_standard_pca_dim3_path = config.save_split_standard_pca_dim3_path
    save_lda_standard_path = config.save_lda_standard_path
    save_split_lda_standard_path = config.save_split_lda_standard_path
    save_split_lda_standard_dim3_path = config.save_split_lda_standard_dim3_path
    save_data_root = config.save_data_root

    # 初步处理
    source_data_dict = read_from_txt(text_list, dataset_root_path)
    # source_data_dict = search_according_shots_num(source_data_dict,
    #                                               config.ways_num,
    #                                               config.information)

    if not split:
        data, attribute = concat_according_profile(source_data_dict, text_list)
        data_dict = {
            "data": data,
            "attribute": attribute[:, :-1]
        }

        # 降维
        data_dict_after_reduce = dim_decay(data_dict, information, method="LDA", standard=standard)

        # 保存数据
        # print(data_dict["data"].shape)
        print(data_dict_after_reduce["data"].shape)
        # with open(save_path, 'wb') as f:
        #     dill.dump(data_dict, f)

        # if standard:
        #     save_name = f"{config.ways_num}ways_LDA_Standard.pkl"
        # else:
        #     save_name = f"{config.ways_num}ways_LDA.pkl"
        #
        # save_path = os.path.join(save_data_root, save_name)
        # with open(save_path, 'wb') as f:
        #     dill.dump(data_dict_after_reduce, f)

        if standard:
            with open(save_lda_standard_path, 'wb') as f:
                dill.dump(data_dict_after_reduce, f)
        else:
            with open(save_lda_path, 'wb') as f:
                dill.dump(data_dict_after_reduce, f)
    else:
        # 将DataFrame数据转化成ndarray数据
        # 并分别根据属性，进行降维
        data = dict()
        data["attribute"] = source_data_dict["profile"].values[:, :-1]
        datas_after_reduce = dict()

        # 确定输出的维度
        if dim3:
            dim_out = 16
        else:
            dim_out = 8

        for key in source_data_dict.keys():
            if key == "profile":
                continue
            data["data"] = source_data_dict[key].values
            data_after_reduce = dim_decay(data, information, dim_out=dim_out, method="LDA", standard=standard)
            datas_after_reduce[key] = data_after_reduce["data"]
            print(data_after_reduce['data'].shape)

        # 创建数据矩阵
        # 其中dim3决定了创建的矩阵是否是3维的
        if not dim3:
            values = []
            for key in datas_after_reduce.keys():
                values.append(datas_after_reduce[key])
            datas = np.concatenate(values, axis=1)
        else:
            values = []
            for i in range(len(data["attribute"])):
                value = []
                for key in datas_after_reduce.keys():
                    data_key_i = np.reshape(datas_after_reduce[key][i], (1, dim_out))
                    print(f"{key}_{datas_after_reduce[key][i].shape}")
                    value.append(data_key_i)
                value = np.concatenate(value, axis=0)[None, ...]
                values.append(value)
            datas = np.concatenate(values, axis=0)

        data_after_reduce_dict = {
            "data": datas,
            "attribute": source_data_dict["profile"].values[:, :-1]
        }

        data_after_reduce_dict["attribute"] = attribute_standard(data_after_reduce_dict["attribute"], information)

        # 保存数据
        print(data_after_reduce_dict["data"].shape)

        if dim3:
            with open(save_split_lda_standard_dim3_path, 'wb') as f:
                dill.dump(data_after_reduce_dict, f)
        else:
            with open(save_split_lda_standard_path, 'wb') as f:
                dill.dump(data_after_reduce_dict, f)


if __name__ == '__main__':
    with open("..\\configs\\config_0.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = EasyDict(config)
    # information = config["information"]
    # text_list = config["text_list"]
    # dataset_root_path = config["dataset_root_path"]
    # origin_data = read_from_txt(text_list, dataset_root_path)
    # data = search_according_shots_num(origin_data, 5, information)
    # print(data["CE"].shape)
    # print(data["profile"])

    # print(origin_data["CE"].values[0])
    # print(origin_data["CP"].values[0])
    # data, attr = concat_according_profile(origin_data, text_list)
    # print(data.shape)
    # print(data[0])
    # save_dict = dict()
    # save_dict["data"] = data
    # save_dict["attribute"] = attr
    # # with open(save_path, 'wb') as f:
    # #     dill.dump(save_dict, f)
    # # with open(save_path, 'rb') as f:
    # #     save_dict = dill.load(f)
    # # print(save_dict)
    # # print(attr)

    process(config, split=False, dim3=False, standard=False)
    path = "..\\processed_data\\5ways_LDA.pkl"
    with open(path, 'rb') as f:
        data = dill.load(f)
    print(np.min(data['data']))
    print(np.max(data['data']))
    print(type(data['attribute']))

    # root = "..\\dataset"
    # test_li = ["SE", "profile"]
    # data_origin = read_from_txt(test_li, root)
    # data = dict()
    # data["attribute"] = data_origin["profile"].values[:, :-1]
    # datas_after_reduce = dict()
    #
    # # 确定输出的维度
    # dim_out = 16
    # # print(data_origin["SE"].values.shape)
    #
    # for key in data_origin.keys():
    #     if key == "profile":
    #         continue
    #     data["data"] = data_origin[key].values
    #     data_after_reduce = dim_decay(data, information, dim_out=dim_out, method="LDA", standard=True)
    #     datas_after_reduce[key] = data_after_reduce["data"]
    #     print(data_after_reduce['data'].shape)

    # # 创建数据矩阵
    # # 其中dim3决定了创建的矩阵是否是3维的
    # values = []
    # for i in range(len(data["attribute"])):
    #     value = []
    #     for key in datas_after_reduce.keys():
    #         data_key_i = np.reshape(datas_after_reduce[key][i], (1, dim_out))
    #         print(f"{key}_{datas_after_reduce[key][i].shape}")
    #         value.append(data_key_i)
    #     value = np.concatenate(value, axis=0)[None, ...]
    #     values.append(value)
    # datas = np.concatenate(values, axis=0)
