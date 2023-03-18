# _*_coding:utf-8_*_
# Author:Zhou JP
# DATE: 16:47 2023/3/15
"""
1. 加载处理后的数据
2. 划分数据集(先不划分数据集，先将扩散模型搭完后再划分数据)
3. 建立映射关系：__getitem__, __len__
"""
import os
from torch.utils.data.dataset import Dataset
import dill


save_path = "..\\processed_data\\data_dict.pkl"
save_lda_path = "..\\processed_data\\data_after_lda.pkl"


class FaultDataset(Dataset):
    def __init__(self):
        super(FaultDataset, self).__init__()
        if not os.path.exists(save_lda_path):
            raise OSError(f"There is not a existed path. Error path: {save_lda_path}")
        with open(save_lda_path, 'rb') as f:
            self.source_data = dill.load(f)
        self.data = self.source_data["data"]
        self.attribute = self.source_data["attribute"]
        self.train_data = None
        self.test_data = None
        self.validation_data = None
        self._len = len(self.data)

    def divide_data(self):
        """将source_data中的数据分成train, validation, test三部分"""
        pass

    def __getitem__(self, item):
        return self.data[item], self.attribute[item]

    def __len__(self):
        return self._len


if __name__ == '__main__':
    dataset = FaultDataset()
    print(dataset.__getitem__(0))
    print(dataset.__len__())
