from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
import csv
import torch
import os
from itertools import chain

import config
from getdata import TestDataset
from network import ConvNet
import torch.nn.functional as F
import time
from alive_progress import alive_bar

model_name = "checkpoint_model_2000.pth"
def predict():
    test_dataset = TestDataset(config.test_data_path)
    model = ConvNet()
    model.eval()
    checkpoint = torch.load(config.model_path + model_name, map_location='cpu')
    model.load_state_dict(checkpoint["model"],strict = False)
    with torch.no_grad():
        outputs = []
        with alive_bar(test_dataset.__len__()) as bar:
            for i in range(test_dataset.__len__()):
#             for i in range(10):
                img = test_dataset.__getitem__(i)
#                 print(img.shape)
                img = img.unsqueeze(0)
                out = model(img)
                num = out.argmax(1)
#                 print(out.shape)
#                 print(num.item())
                outputs.append(num.item())
                #显示进度
                bar()
        result = pd.DataFrame({"ImageId":np.arange(1,len(outputs)+1),
                               "Label":outputs})
        result.to_csv(config.submission_path+"result.csv",index=False)
if __name__ == '__main__':
    predict()


