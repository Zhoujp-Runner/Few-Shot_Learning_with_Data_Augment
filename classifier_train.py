# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 15:33 2023/5/21
import os
import torch
import yaml

from easydict import EasyDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from visdom import Visdom

from models.model import GuidedClassifier
from models.diffusion import DiffusionModel
from process_data.dataset import GuidedDataset


"""
dataset_type: TEP

guided v1.0
hidden 256 > 128 > 64 > 32
best performance loss = 2.75 - 2.8

hidden 256
dataset z score standard
best performance loss = 1.6 - 1.7
"""

"""
dataset_type: Hydraulic

guided v1.0
hidden 256 > 128 > 64
best performance loss = 0.5 - 1.0 closer to 0.5
"""


def main():
    dataset_type = 'Hydraulic'
    path = f"configs\\config_0.yaml"
    config = load_config(path)
    classifier_path = config.classifier_path

    dataset = GuidedDataset(config, dataset_type)
    dim_in = dataset.train_data.shape[-1] - 1
    if dataset.dataset_type == 'TEP':
        dim_out = 21  # TEP数据集一共有21种故障状态
    elif dataset.dataset_type == 'Hydraulic':
        dim_out = 144  # 液压系统数据集一共有144种故障状态
    else:
        raise ValueError("No such dataset type!")
    dim_hidden = 256
    batch_size = 64
    epochs = 2000
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    diffusion = DiffusionModel(config)
    classifier = GuidedClassifier(dim_in=dim_in,
                                  dim_hidden=dim_hidden,
                                  dim_out=dim_out,
                                  diffusion_num_step=config.num_diffusion_steps)
    classifier = classifier.to(diffusion.device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    viz = Visdom(env="Guided Classifier")

    min_loss = 100
    min_epoch = 0
    min_classifier_dict = None
    opt_state_dict = None

    for epoch in range(epochs):
        pb_dataloader = tqdm(dataloader, desc=f"Epoch{epoch}: ")
        total_loss = 0
        for batch, label in pb_dataloader:
            optimizer.zero_grad()

            size = batch.shape[0]
            batch = batch.to(diffusion.device)
            time_steps = diffusion.sample_t(size)
            noise = torch.randn_like(batch)
            x_t = diffusion.diffusion_at_time_t(batch, time_steps, noise)
            label = label_to_onehot(label, dim_out).to(diffusion.device)

            out = classifier(x_t, time_steps)

            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pb_dataloader.set_postfix_str(f"Loss={loss.item()}")

            if loss < min_loss:
                min_classifier_dict = classifier.state_dict()
                opt_state_dict = optimizer.state_dict()
                min_epoch = epoch
        loss_mean = total_loss / len(dataloader)
        viz.line(X=[epoch], Y=[loss_mean], win="mean loss", update='append')

    classifier_root = r"experiments\\models\\classifier"
    classifier_name = f"Hydraulic_classifier_zscore_dimhidden{dim_hidden}.pkl"
    save_path = os.path.join(classifier_root, classifier_name)
    if min_classifier_dict is not None:
        check_point = {
            "model": min_classifier_dict,
            "optimizer": opt_state_dict,
            "epoch": min_epoch
        }
        torch.save(check_point, save_path)


def load_config(path):
    if not os.path.exists(path):
        raise ValueError("No such path!")
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)
    return config


def label_to_onehot(label, dim_out):
    """针对TEP数据集"""
    batch_size = label.shape[0]
    onehot = torch.zeros(batch_size, dim_out)
    for index, y in enumerate(label):
        onehot[index][y - 1] = 1
    return onehot


if __name__ == '__main__':
    main()
