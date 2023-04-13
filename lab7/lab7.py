# init block
# %%
import torch
import os
import matplotlib.pyplot as plt
import sys
if 'lab7_model' in sys.modules:
    del sys.modules['lab7_model']
if 'lab7_data' in sys.modules:
    del sys.modules['lab7_data']
from lab7_model import RNNModel as Model
from lab7_data import LoadData

batch = 100
hiden_dim = 8
layer_dim = 3
epoches = 100
steps = 150


dataset_train, dataset_test = LoadData(batch, 10)
x, y = next(iter(dataset_train))

    
model = Model(x.size(2), hiden_dim, layer_dim, y.size(1))
model.info()


# train block
# %%
# model.load()
# cuda
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: %s" % device)
model = model.to(device)

print("Train")
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# https://pytorch.org/docs/stable/nn.html#loss-functions
loss_fn = torch.nn.MSELoss(reduction="mean")
for epoche in range(epoches):
    err = 0.
    for step in range(steps):
        inputs, labels = next(iter(dataset_train))
        
        # normalizacja
        inputs = inputs / torch.Tensor([110.,12.])
        labels = labels / torch.Tensor([110.])

        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        outputs = model(inputs)
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()
        # Model parameters optimization
        optimizer.step()
        
        err += loss.item()
        print("\rerror = %f "%(err), end="")        
    print("\repoch= %d error= %f: \n"%(epoche,err/step))        
model.save()
