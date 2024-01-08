# init block
# %%
import torch
import os
import matplotlib.pyplot as plt
import sys
if 'lab4_model' in sys.modules: del sys.modules['lab4_model']
from lab4_model import ConvModel as Model
if 'lab4_data' in sys.modules: del sys.modules['lab4_data']
from lab4_data import DataSet, show_images, prepare_edge

in_size = 1
model = Model(x=6, out_size=1, in_size=in_size)
model.info()
# model.load()


batch = 10
dataset_test = DataSet(batch=batch)
# dataset_train = dataset_test
p = os.path.join('lab4_dataset', 'train', '*.png')
dataset_train = DataSet(path=p, batch=batch, div=False, aug=True)
loss_fn = torch.nn.MSELoss(reduction="mean")

# train block
# %%
# cuda
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: %s" % device)
model = model.to(device)

print("Train")
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

for epoche in range(7):
    err = 0
    true_error = 0
    for step in range(150):
        x, y_true = next(iter(dataset_train))

        x = x.to(device)
        y_true = y_true.to(device)
        
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        optimizer.step()

        err += loss.item()
        true_error += loss_fn(y_pred, y_true).item()
        print("\rerror = %f , real= %f" % (err, true_error), end="")
    print("\repoch= %d error= %f, real=%f" % (epoche, err, true_error))
    show_images(model.conv1.weight.clone().cpu().detach())
    show_images(x[:2].cpu())
    show_images(y_true[:2].cpu(), y_pred[:2].cpu().detach())
    
    
model.save()

# eval on dataset data
# %%
model.load()
model.eval()
# pred on dataset
x, y_true = next(iter(dataset_test))
x = x.to(device)
y_pred = model(x).cpu().detach()
eval_error = loss_fn(y_pred, y_true).item()
print("eval_dataset_error: %f" % eval_error)

print("Y from dataset (ground truth)")
show_images(x[:1].cpu(),y_true[:1])
print("Y from model (predicted)")
show_images(x[:1].cpu(), y_pred[:1] )
print("difference  abs( (ground truth) - (predicted) )")
tmp = torch.abs(y_true[:1]-y_pred[:1])
show_images(tmp, vmax=tmp.max())
t1,t2 = prepare_edge(x[:1].cpu())
print("Y from edge detection fun")
show_images(t1,t2)

# %%
x.size()
