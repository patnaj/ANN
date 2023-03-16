# %%
from typing import Iterator
import torch
import os
import glob
import torchvision
import numpy as np
import matplotlib.pyplot as plt


# https://www.kaggle.com/code/riyueguanghua/anime-sketch-colorization/

# dataset = datasets.ImageFolder('path/to/data', transform=transform)
path = os.path.join('lab4_dataset', 'test', '*.png')
files = glob.glob(path)
images = torch.cat([
    torchvision.io.read_image(
        p, torchvision.io.ImageReadMode.GRAY).unsqueeze(0)
    for p in files], dim=0)

# %%


def prepare_edge(data):
    f_x = torch.Tensor([[[[-1., 0., 1.]]]]).expand(1, 1, 3, 3)
    f_y = torch.Tensor([[[[-1.], [0.], [1.]]]]).expand(1, 1, 3, 3)
    filter = torch.cat([f_x, f_y])
    y = torch.nn.functional.pad(data, pad=(1, 1, 1, 1), mode='reflect')
    y = torch.nn.functional.conv2d(y, filter, padding=0)
    y = (y*y).sum(dim=1, keepdim=True)
    # y = torch.nn.functional.normalize(y, dim=1)
    mi = y.min()
    ma = y.max()
    y = (y / (ma-mi)) - mi
    return data, y


def prepare_divaide(data):
    s = data.size()
    data_x = data[:, :, :, :(s[3] >> 1)]
    data_y = data[:, :, :, (s[3] >> 1):]
    return data_x, 1. - data_y


def show_images(x, y=None, vmin=None, vmax=None):
    xd = x.permute(0, 2, 3, 1)
    n = xd.size(0)
    if y is not None:
        yd = y.permute(0, 2, 3, 1)
        f = plt.figure()
        for i in range(min(n, 10)):
            f.add_subplot(n, 2, (i*2)+1)
            plt.imshow(xd[i], cmap='gray', vmin=vmin, vmax=vmax)
            f.add_subplot(n, 2, i*2+2)
            plt.imshow(yd[i], cmap='gray', vmin=vmin, vmax=vmax)
        plt.show()
    else:
        f = plt.figure()
        for i in range(min(n, 10)):
            f.add_subplot(n, 2, i+1)
            plt.imshow(xd[i], cmap='gray', vmin=vmin, vmax=vmax)
        plt.show()





# %%

defpath = os.path.join('lab4_dataset', 'test', '*.png')


class DataSet():

    def __init__(self, batch=10, path=defpath, div=True, aug = False):
        self.batch = batch
        self.aug = aug
        self.div = div

        files = glob.glob(path)
        images = torch.cat([
            torchvision.io.read_image(
                p, torchvision.io.ImageReadMode.GRAY).unsqueeze(0)
            for p in files], dim=0)

        self.images = images / 255
        # self.x, self.y = prepare_divaide(
        #     self.images) if div else prepare_edge(self.images)
        self.x, self.y = prepare_divaide(
            self.images) if div else (self.images, self.images)
            

        # self.rot = torchvision.transforms.

    def augment(self,x, y):
        if self.aug:
            rot = int(torch.rand(1)*180)
            # idx = (torch.rand(self.batch)*n).long()    
            x = torchvision.transforms.functional.rotate(x, rot, fill=1.)
            y = torchvision.transforms.functional.rotate(y, rot)
            
        if not self.div:
            _, y = prepare_edge(y)
        return x, y

    def next_batch(self):
        n = self.x.size(0)
        idx = (torch.rand(self.batch)*n).long()
        # x, y = self.x[idx], self.y[idx]
        yield self.augment( self.x[idx], self.y[idx] )
        
    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter(self.next_batch())


if __name__ == '__main__':
    p = os.path.join('lab4_dataset', 'train', '*.png')
    ds = DataSet(batch=4, path=p, div=False, aug=True)
    x, y = next(iter(ds))
    
    s = next(iter(ds))
    show_images(x,y)
    self =ds

    ds.x.size()
    ds.y.size()

    ds.x.size()


    x, y = prepare_divaide(images/255)
    show_images(x, y)
    _, y2 = prepare_edge(x)
    show_images(y, y2)
    
    ds = DataSet(batch=4, aug=True)
    x, y = next(iter(ds))
    show_images(x,y)

    # %%
