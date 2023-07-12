import torch
from torchvision import transforms
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import config

class TrainDataset(Dataset):
    def __init__(self,train_data_path):
        data_df = pd.read_csv(train_data_path)
        label_df = data_df["label"]
        feature_df = data_df.drop("label", axis=1)
        # 归一化处理
        feature_df = feature_df/255.0
        # 进行数据变换，变换成1*28*28(C*H*W)的图像输入形式
        feature_df = feature_df.apply(lambda x:x.values.reshape(1,28,28), axis=1)
        self.label_df = label_df
        self.images = feature_df
        self.transform = transforms.ToTensor()
    def __len__(self):
        return len(self.label_df)
    def __getitem__(self,index):
        return self.label_df[index],self.transform(self.images[index]).reshape(1,28,28).float()
class TestDataset(Dataset):
    def __init__(self,test_data_path):
        feature_df = pd.read_csv(test_data_path)
        feature_df = feature_df/255.0
        self.images = feature_df.apply(lambda x:x.values.reshape(1,28,28), axis=1)
        self.transform = transforms.ToTensor()
    def __len__(self):
        return len(self.images)
    def __getitem__(self,index):
        return self.transform(self.images[index]).reshape(1,28,28).float()
        