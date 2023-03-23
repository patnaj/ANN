# init block
# %%
import torch
import os
import matplotlib.pyplot as plt
import sys
if 'lab5_model' in sys.modules:
    del sys.modules['lab5_model']
if 'lab5_data' in sys.modules:
    del sys.modules['lab5_data']
from lab5_model import ConvModel as Model
from lab5_data import LoadData, show_images

in_ch = 1  # one chanel - grey image
cla = 10  # MNIST 10 klass [0,1,2,3,4,5,6,7,8,9]

model = Model(in_ch, 16, cla)
# model.load()


batch = 100
dataset_train, dataset_test = LoadData(batch, 1)

# train block
# %%
# cuda
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: %s" % device)
model = model.to(device)

print("Train")
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# https://pytorch.org/docs/stable/nn.html#loss-functions
loss_fn = torch.nn.CrossEntropyLoss()
for epoche in range(7):
    err = 0.
    for step in range(150):
        x, y_true = next(iter(dataset_train))

        x = x.to(device)
        y_true = y_true.to(device)

        optimizer.zero_grad()
        y_pred = model(x)
        
        # this is fullyconv model so from input 27x27 we have output 3x3,
        # but y is 1 so we expand y
        s=y_pred.size()
        y2_true = y_true.unsqueeze(1).unsqueeze(1).expand(-1,s[2], s[3])
        
        # print(y_pred.size(), y_true.size())
        loss = loss_fn(y_pred, y2_true)
        loss.backward()
        optimizer.step()

        err += loss.item()
        print("\rloss_sum = %f" % (err), end="")
    print("\repoch= %d error= %f" % (epoche, err/(batch*3*3)))
    show_images(model.block1.conv1.weight.cpu().detach(), isimg=False)
    show_images(x[:2])
    # pred per class
    y = y_pred[:2].max(dim=2)[0].max(dim=2)[0]
    print("per class", y)
    # pred class
    yc = y.max(dim=1)[1]
    print("pred class", yc)
    print("true class", y_true[:2])
model.save()

# eval on dataset data
# %%
model.load()
model.eval()
# pred on dataset
x, y_true = next(iter(dataset_test))
x = x.to(device)
y_pred = model(x).cpu().detach()
x = x.cpu()
show_images(x[:1])
print("pred cls:");
print(y_pred[:1].max(dim=1)[1])
print("pred streng:");
print(y_pred[:1].max(dim=1)[0])

# %%
model = model.cpu()
x, y_true = next(iter(dataset_test))
xx = torch.zeros(1,1, 300,300)
off = (torch.rand(2)*(300-28)).int()
xx[0,0,off[0]:off[0]+28, off[1]:off[1]+28] = x[0,0,:,:]
show_images(xx[:1])
y_pred = model(xx).detach()

print("pred cls:");
print(y_pred[:1].max(dim=1)[1])
print("pred streng:");
print(y_pred[:1].max(dim=1)[0])

show_images(xx[:1])
show_images(y_pred[:1].max(dim=1)[0].unsqueeze(1))

# %%
