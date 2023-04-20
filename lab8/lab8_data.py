# %%
from typing import Iterator
import torch
import os
import glob
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# download trainng.zip into lab8 project
# https://www.kaggle.com/competitions/facial-keypoints-detection/data?select=training.zip

# example
# https://www.kaggle.com/code/ready234/facial-detection


class DataSet():
    def __init__(self, data_x, data_y, batch=10):
        self.data_x = data_x
        self.data_y = data_y
        self.batch = batch

    def augment(self, x, y):
        return x, y

    def next_batch(self):
        n = self.data_x.size(0)
        idx = (torch.rand(self.batch)*n).long()
        
        yield self.augment(self.data_x[idx], self.data_y[idx])

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter(self.next_batch())


df = None
# data = None
default_params = [
    "left_eye_center_x",
    "left_eye_center_y",
    "right_eye_center_x",
    "right_eye_center_y",
    "mouth_left_corner_x",
    "mouth_left_corner_y",
    "mouth_right_corner_x",
    "mouth_right_corner_y"
]


def LoadData(batch_size_train=10, batch_size_test=10, params=default_params, train_size=0.7):
    global df
    if df is None:
        print("load: './training.zip'")
        df = pd.read_csv('./training.zip',
                         low_memory=False)[[*params, "Image"]].dropna()
        df.info()
        # to avoid scientific notation
        pd.options.display.float_format = '{:.2f}'.format

        image_x = np.array([np.array([float(i) for i in arr.split(" ")]).reshape(96,96,1)  for arr in df["Image"]])
        image_x = torch.tensor(image_x, dtype=torch.float32) / 255.
        
        data_y = torch.tensor(df[params].to_numpy())

    idx_train = (torch.rand(int(image_x.size(0) * train_size))
                 * image_x.size(0)).long()
    idx_test = torch.Tensor([True]).bool().repeat(image_x.size(0))
    idx_test[idx_train] = False

    return DataSet(image_x[idx_train, :], data_y[idx_train, :],
                   batch_size_train), DataSet(image_x[idx_test, :], data_y[idx_test, :], batch_size_test)


if __name__ == '__main__':
    train, test = LoadData(3, 3)
    x, y = next(iter(train))
    
    xy = zip(default_params ,y[1])
    print(list(xy))
    
    # img = plt.imshow(x[1])
    # for i in range(y[1].size(0)>>1):
    #     plt.plot(y[i*2],y[i*2+1],'r*')