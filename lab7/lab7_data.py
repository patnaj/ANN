# %%
from typing import Iterator
import torch
import os
import glob
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# https://www.kaggle.com/datasets/sudalairajkumar/daily-temperature-of-major-cities
# https://www.kaggle.com/code/damodarabarbosa/daily-temperature-analysis-and-visualization

# https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html 
# https://towardsdatascience.com/pytorch-basics-how-to-train-your-neural-net-intro-to-rnn-cb6ebc594677
# https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/
# https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
# https://pytorch.org/docs/1.13/generated/torch.nn.RNN.html#torch.nn.RNN
# https://pytorch.org/docs/1.13/generated/torch.nn.LSTM.html#torch.nn.LSTM


class DataSet():
    def __init__(self, data, batch=10, x_num=6, y_num=1):
        self.data = data
        self.batch = batch
        self.x_num = x_num
        self.y_num = y_num

    def augment(self, x, y):
        return x, y

    def next_batch(self):
        s = (self.x_num+self.y_num)
        n = self.data.size(0) - s
        idx = (torch.rand(self.batch)*n).long()
        idx = idx.unsqueeze(1).expand(-1, s) + torch.arange(0, s)
        d = self.data[idx]
        yield self.augment(d[:, :self.x_num, :], d[:, self.x_num:, :])

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter(self.next_batch())

df = None
# data = None

def LoadData(batch_size_train=10, batch_size_test=10, country='Poland', xnum=6, ynum=1):
    global df
    if df is None:
        print("load: './archive.zip'")
        df = pd.read_csv('./archive.zip', low_memory=False)
    df.info()
    # to avoid scientific notation
    pd.options.display.float_format = '{:.2f}'.format
    df.describe()

    data = df.query('Country=="%s"' % (country))
    data.groupby('Country').describe()

    test = data.query("Year <= %f" % data['Year'].median())
    train = data.query("Year > %f" % data['Year'].median())
    test_pt = torch.Tensor(test[['AvgTemperature', 'Month']].to_numpy())
    train_pt = torch.Tensor(train[['AvgTemperature', 'Month']].to_numpy())

    # debug
    # self = DataSet(train_pt)

    return DataSet(train_pt, batch_size_train), DataSet(test_pt, batch_size_test)


if __name__ == '__main__':
    train, test = LoadData(3, 3)
    x, y = next(iter(train))    
    plt.plot(torch.arange(train.x_num), x[0,:,0],'gs'
             ,torch.arange(train.y_num)+train.x_num,y[0,:,0],'bs')
