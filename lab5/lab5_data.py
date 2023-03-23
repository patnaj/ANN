# %%
from typing import Iterator
import torch
import os
import glob
import torchvision
import numpy as np
import matplotlib.pyplot as plt


def LoadData(batch_size_train = 10, batch_size_test = 10):

    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files/', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
    batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
    batch_size=batch_size_test, shuffle=True)
    
    return train_loader, test_loader


def show_images(x, y=None, isimg = True ):
    vmin, vmax, cmap = (0.,1.,'gray') if isimg else (None, None, None)
    xd = x.permute(0, 2, 3, 1)
    n = xd.size(0)
    if y is not None:
        yd = y.permute(0, 2, 3, 1)
        f = plt.figure()
        for i in range(min(n, 10)):
            f.add_subplot(n, 2, (i*2)+1)
            plt.imshow(xd[i], cmap=cmap, vmin=vmin, vmax=vmax)
            f.add_subplot(n, 2, i*2+2)
            plt.imshow(yd[i], cmap=cmap, vmin=vmin, vmax=vmax)
        plt.show()
    else:
        f = plt.figure()
        for i in range(min(n, 10)):
            f.add_subplot(n, 2, i+1)
            plt.imshow(xd[i], cmap=cmap, vmin=vmin, vmax=vmax)
        plt.show()



if __name__ == '__main__':
    train , test = LoadData(3,3)
    x, y = next(iter(train))
    show_images(x)
