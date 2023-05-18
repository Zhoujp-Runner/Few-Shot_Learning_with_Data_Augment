# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 16:39 2023/4/5
"""
在二维平面内，数据分布的可视化
"""
import dill
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import torch
import yaml
from easydict import EasyDict
from itertools import product
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, MDS

from process_data.dataset import FaultDataset, TEPDataset
from process_data.analysis import attribute_standard, transform_attribute_to_label, information_standard


def plot_points(data, ax, color=('blue', 'red')):
    """
    绘制增强前后数据的分布
    :param data: [data_source, data_augment]
    :param ax: fig.axes
    :param color: 数据的颜色，默认蓝色为真实数据，红色为生成数据
    """
    size = len(data)
    if size != 2:
        raise ValueError("Except data of size(2)")

    data_source = data[0]
    data_augment = data[1]

    data_source_x = data_source[:, 0]
    data_source_y = data_source[:, 1]

    data_source_color = color[0]

    ax.scatter(data_source_x, data_source_y, c=data_source_color)
    if len(data_augment) != 0:
        data_augment_x = data_augment[:, 0]
        data_augment_y = data_augment[:, 1]
        data_augment_color = color[1]
        ax.scatter(data_augment_x, data_augment_y, c=data_augment_color)


def unranked_unique(arr):
    """
    由于np.unique会自动排序，所以这里写一个不会自动排序的unique()
    :param arr: 输入矩阵
    :return: 保留顺序的unique矩阵
    """
    _, idx = np.unique(arr, return_index=True)
    return arr[np.sort(idx)]


if __name__ == '__main__':
    # with open("..\\configs\\config_0.yaml") as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    #
    # config = EasyDict(config)
    #
    # # a = [1, 1]
    # # b = [1, 2]
    # cooler, valve, pump, hydraulic = config.information
    # att_iter = product(cooler, valve, pump, hydraulic)
    # att_list = []
    # for item in att_iter:
    #     att_list.append(list(item))
    # att_list = attribute_standard(att_list, config.information)
    # att_list = torch.FloatTensor(att_list)
    # # print(len(att_list))
    #
    # dataset = FaultDataset(config, method=config.method)
    # dataset_aug = FaultDataset(config, method=config.method, augment=True, with_train=False)
    # dataset_aug_with_train = FaultDataset(config, method=config.method, augment=True, with_train=True)
    # # print(dataset_aug.test_data.shape)
    #
    # colors = ['blue', 'red', 'green', 'purple']
    #
    # # source = []
    # # source2 = []
    # # augment = []
    # source_index = []
    # label_source = []
    # label_source2 = []
    # label_augment = []
    # i = 0
    # fig, axes = plt.subplots()
    # # for i, item in enumerate(att_list):
    # #     num = 0
    # #     for idx, att in enumerate(dataset.train_attribute):
    # #         # print(item)
    # #         # print(att)
    # #         if torch.equal(item, att):
    # #             source.append(dataset.train_data[idx].numpy())
    # #             source_index.append(idx)
    # #             num += 1
    # #         # elif torch.equal(item, att) and num >= 4:
    # #         #     source2.append(dataset.train_data[idx].numpy())
    # #     label_source.append([i] * 4)
    # #     # label_source2.append([i] * 4)
    # #     for idx_, att_ in enumerate(dataset_aug.train_attribute):
    # #         if torch.equal(item, att_) and idx_ not in source_index:
    # #             augment.append(dataset_aug.train_data[idx_].numpy())
    # #     label_augment.append([i] * 20)
    # #     num += 1
    # #     if num == 5:
    # #         break
    #
    # # label_source = np.reshape(np.array(label_source), -1)
    # # label_source2 = np.copy(label_source)
    # # label_augment = np.reshape(np.array(label_augment), -1)
    # # # print(np.argwhere(source == 143.))
    # # print(len(label_source))
    # #
    #
    # lda = LinearDiscriminantAnalysis(n_components=2)
    # tsne = TSNE(n_components=2, perplexity=3)
    # mds = MDS(n_components=2)
    #
    # # # lda
    # # source_data = dataset.train_data.numpy()
    # # augment_data = dataset_aug.train_data.numpy()
    # # # augment_data = []
    # # # for i in range(5):
    # # #     augment_data.append(augment[20*i: 9+20*i])
    # # # augment_data = np.concatenate(augment_data, axis=0)
    # # train_data = np.concatenate([source_data, augment_data], axis=0)
    # # print(train_data.shape)
    # # information = config.information
    # # information = information_standard(information)
    # # source_attribute = dataset.train_attribute.numpy()
    # # augment_attribute = dataset_aug.train_attribute.numpy()
    # # augment_attribute_list = []
    # # # for i in range(5):
    # # #     augment_attribute_list.append(augment_attribute[20*i: 9+20*i])
    # # # augment_attribute = np.concatenate(augment_attribute_list, axis=0)
    # # attribute = np.concatenate([source_attribute, augment_attribute], axis=0)
    # # label = transform_attribute_to_label(attribute, information)
    # # print(label.shape)
    # # # data_viz = tsne.fit_transform(train_data)
    # # data_viz = lda.fit_transform(train_data, label)
    # # source = data_viz[:45, :]
    # # augment = data_viz[45:, :]
    # # print(source.shape)
    # # print(augment.shape)
    #
    # # tsne
    # source_data = dataset.train_data.numpy()
    # augment = dataset_aug.train_data.numpy()
    # print(source_data)
    # # print(augment)
    # augment_data = []
    # # source_data = []
    # for i in range(5):
    #     augment_data.append(augment[20*i: 9+20*i])
    # # for i in range(5):
    # #     source_data.append(augment[9*i: 9+9*i])
    # augment_data = np.concatenate(augment_data, axis=0)
    # # source_data = np.concatenate(source_data, axis=0)
    # # print(augment_data.shape)
    # train_data = np.concatenate([source_data, augment_data], axis=0)
    # data_viz = tsne.fit_transform(train_data)
    # source = data_viz[:45, :]
    # augment = data_viz[45:, :]
    #
    # # # mds
    # # source_data = dataset.train_data.numpy()
    # # augment = dataset_aug.train_data.numpy()
    # # print(source_data)
    # # print(augment)
    # # augment_data = []
    # # for i in range(5):
    # #     augment_data.append(augment[20*i: 9+20*i])
    # # augment_data = np.concatenate(augment_data, axis=0)
    # # # print(augment_data.shape)
    # # train_data = np.concatenate([source_data, augment_data], axis=0)
    # # data_viz = mds.fit_transform(train_data)
    # # source = data_viz[:45, :]
    # # augment = data_viz[45:, :]
    #
    # data = [source, augment]
    # # data = [source, source2]
    # plot_points(data, axes)
    # plt.show()
    #
    #
    # # fig, axes = plt.subplots()
    # # for i in range(2):
    # #     source = []
    # #     augment = []
    # #     for idx, att in enumerate(dataset.train_attribute):
    # #         # print(item)
    # #         # print(att)
    # #         if torch.equal(att_list[i], att):
    # #             source.append(dataset.train_data[idx].numpy())
    # #             source_index.append(idx)
    # #     for idx_, att_ in enumerate(dataset_aug.train_attribute):
    # #         if torch.equal(att_list[i], att_) and idx_ not in source_index:
    # #             augment.append(dataset_aug.train_data[idx_].numpy())
    # #
    # #     pca = PCA(n_components=2)
    # #     source = np.array(source)
    # #     augment = np.array(augment)
    # #     print(source)
    # #     print(augment)
    # #     source = pca.fit_transform(source)
    # #     augment = pca.fit_transform(augment)
    # #     print(source.shape)
    # #     print(augment.shape)
    # #     data = [source, augment]
    # #     plot_points(data, axes, color=[colors[2*i], colors[2*i+1]])
    # # plt.show()
    #
    # with open("..\\processed_data\\save_lda_standard.pkl", 'rb') as f:
    #     data = dill.load(f)
    # print(data["data"])
    #
    # with open("..\\processed_data\\data_after_lda.pkl", 'rb') as f:
    #     data2 = dill.load(f)
    # print(data2["data"])

    with open("..\\configs\\config_0.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = EasyDict(config)

    # a = [1, 1]
    # b = [1, 2]
    cooler, valve, pump, hydraulic = config.information
    att_iter = product(cooler, valve, pump, hydraulic)
    att_list = []
    for item in att_iter:
        att_list.append(list(item))
    att_list = attribute_standard(att_list, config.information)
    att_list = torch.FloatTensor(att_list)
    # print(len(att_list))

    dataset = FaultDataset(config, method=config.method)
    dataset_aug = FaultDataset(config, method=config.method, augment=True, with_train=False)
    dataset_aug_with_train = FaultDataset(config, method=config.method, augment=True, with_train=True)
    # print(dataset_aug.test_data.shape)

    colors = ['blue', 'red', 'green', 'purple']

    # source = []
    # source2 = []
    # augment = []
    source_index = []
    label_source = []
    label_source2 = []
    label_augment = []
    i = 0
    fig, axes = plt.subplots(5, 6)

    mds = MDS(n_components=2)

    # # tsne
    # source_data = dataset.train_data.numpy()
    # augment = dataset_aug.train_data.numpy()
    # print(source_data)
    # # print(augment)
    # augment_data = []
    # # source_data = []
    # for i in range(5):
    #     augment_data.append(augment[20*i: 9+20*i])
    # # for i in range(5):
    # #     source_data.append(augment[9*i: 9+9*i])
    # augment_data = np.concatenate(augment_data, axis=0)
    # # source_data = np.concatenate(source_data, axis=0)
    # # print(augment_data.shape)
    # train_data = np.concatenate([source_data, augment_data], axis=0)
    # data_viz = tsne.fit_transform(train_data)
    # source = data_viz[:45, :]
    # augment = data_viz[45:, :]
    #
    # data = [source, augment]
    # # data = [source, source2]
    # plot_points(data, axes)
    # plt.show()

    with open("..\\processed_data\\test\\augment\\11_TEP_60.pkl", 'rb') as f:
        datas = dill.load(f)
    with open("..\\processed_data\\tep_train_lda_standard.pkl", 'rb') as f:
        source_datas = dill.load(f)
    batch_size = datas.shape[0]
    datas = np.reshape(datas, (batch_size, 17))
    data, label = np.split(datas, [16, ], axis=-1)
    print(data.shape)
    print(label)

    # source_data, source_label = np.split(source_datas, [52, ], axis=-1)
    # source_label = np.squeeze(source_label, axis=-1)
    # lda = LinearDiscriminantAnalysis(n_components=16)
    # source_data = lda.fit_transform(source_data, source_label)
    # source_label = source_label[..., None]
    # source_datas = np.concatenate([source_data, source_label], axis=-1)

    # label = np.unique(label)
    label = unranked_unique(label)
    print(label)
    expect_datas = []
    expect_labels = []
    index = np.arange(0, 480)
    for y in label:
        data_list = []
        for item in source_datas:
            if item[-1] == y:
                item = item[:-1]
                item = item[None, ]
                data_list.append(item)
        expect_data = np.concatenate(data_list, axis=0)
        indices = np.random.choice(index, 60)
        labels = np.array([y]*60)
        expect_labels.append(labels)
        expect_datas.append(expect_data[indices])
        # expect_datas.append(expect_data)
    expect_datas = np.concatenate(expect_datas, axis=0)
    expect_labels = np.concatenate(expect_labels, axis=0)
    print(expect_datas.shape)
    print(expect_labels)

    for i in range(1, 31):
        tsne = TSNE(n_components=2, perplexity=i)
        data_viz = tsne.fit_transform(expect_datas)
        # x0 = data_viz[:480, 0]
        # y0 = data_viz[:480, 1]
        # x1 = data_viz[480:960, 0]
        # y1 = data_viz[480:960, 1]
        # x2 = data_viz[960:1440, 0]
        # y2 = data_viz[960:1440, 1]
        # x3 = data_viz[1440:1920, 0]
        # y3 = data_viz[1440:1920, 1]
        # x4 = data_viz[1920:2400, 0]
        # y4 = data_viz[1920:2400, 1]
        # x0 = data_viz[:9, 0]
        # y0 = data_viz[:9, 1]
        # x1 = data_viz[9:18, 0]
        # y1 = data_viz[9:18, 1]
        # x2 = data_viz[18:27, 0]
        # y2 = data_viz[18:27, 1]
        # x3 = data_viz[27:36, 0]
        # y3 = data_viz[27:36, 1]
        # x4 = data_viz[36:45, 0]
        # y4 = data_viz[36:45, 1]
        x0 = data_viz[:60, 0]
        y0 = data_viz[:60, 1]
        x1 = data_viz[60:120, 0]
        y1 = data_viz[60:120, 1]
        x2 = data_viz[120:180, 0]
        y2 = data_viz[120:180, 1]
        x3 = data_viz[180:240, 0]
        y3 = data_viz[180:240, 1]
        x4 = data_viz[240:300, 0]
        y4 = data_viz[240:300, 1]
        # x0 = data_viz[:300, 0]
        # y0 = data_viz[:300, 1]
        # x1 = data_viz[300:600, 0]
        # y1 = data_viz[300:600, 1]
        # x2 = data_viz[600:900, 0]
        # y2 = data_viz[600:900, 1]
        # x3 = data_viz[900:1200, 0]
        # y3 = data_viz[900:1200, 1]
        # x4 = data_viz[1200:1500, 0]
        # y4 = data_viz[1200:1500, 1]
        index = i - 1
        row_index = index // 6
        col_index = index % 6
        axes[row_index][col_index].scatter(x0, y0, c='black')
        axes[row_index][col_index].scatter(x1, y1, c='green')
        axes[row_index][col_index].scatter(x2, y2, c='blue')
        axes[row_index][col_index].scatter(x3, y3, c='red')
        axes[row_index][col_index].scatter(x4, y4, c='purple')
    plt.show()

