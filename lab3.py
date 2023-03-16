# init block
# %%
import torch
import matplotlib.pyplot as plt
from lab2_model import TinyModel as Model

input_range = 1
model = Model(1024, input_range=input_range)
model.load()


def data_plot(title, x, y, y2=None):
    fig, ax = plt.subplots()
    x = x.view(-1)
    ax.plot(x, y.view(-1), 'o')
    if y2 is not None:
        ax.plot(x, y2.view(-1), 'o')
    ax.legend(title)
    plt.show()


# function generator block
# %%
def data_fun(batch=30, range=2*torch.pi):
    x = torch.rand(batch, input_range)*range
    return x, (.2*torch.sin(2*x))+torch.tanh(x)


# test
x, y = data_fun(1500)
print("x=", x)
print("y=", y)
data_plot(["Fun test"], x, y)


# dataset init block
# %%


# prepare training dataset, add noise and errors
def dataset_gen(batch=500, range=2*torch.pi, noise=True, errors=True):
    x, y_true = data_fun(batch)
    # add noise
    y_real = y_true.clone()
    if noise:
        y_real += torch.rand(batch, 1)*0.3-0.15
    # add errors
    y_real2 = y_real.clone()
    if errors:
        num = .15
        idx = (torch.rand(int(batch*num))*batch).long()
        y_real2[idx, :] += .1+torch.rand(int(batch*num), 1)*0.6
        idx = (torch.rand(int(batch*num))*batch).long()
        y_real2[idx, :] -= .1+torch.rand(int(batch*num), 1)*0.6
    return x, y_real2, y_true


# dataset
# dataset = dataset_gen(noise=False,errors=False)
dataset = dataset_gen(200)
data_plot(["Real data", "Fun data"], *dataset)


# data random batch
def data_gen(dataset, batch=30):
    size = dataset[0].size(0)
    idx = (torch.rand(batch)*size).long()
    return dataset[0][idx], dataset[1][idx], dataset[2][idx]


# batch sample
data_plot(["Batch sample", "Fun"], *data_gen(dataset))

# train block
# %%
# cuda
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: %s" % device)
model = model.to(device)

print("Train")
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss(reduction="mean")
for epoche in range(30):
    err = 0
    true_error = 0
    for step in range(50):
        # Every data instance is an input + label pair
        inputs, labels, ground_true = data_gen(dataset, 100)

        inputs = inputs.to(device)
        labels = labels.to(device)
        ground_true = ground_true.to(device)
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
        true_error += loss_fn(outputs, ground_true).item()
        print("\rerror = %f , real= %f" % (err, true_error), end="")
    print("\repoch= %d error= %f, real=%f" % (epoche, err, true_error))
model.save()

# eval on dataset data
# %%
model.eval()
# pred on dataset
x, y_dataset, y_true = dataset
y_pred = model(x)
eval_error = loss_fn(y_pred, y_true).item()
print("eval_dataset_error: %f" % eval_error)
data_plot(["Pred(dataset_x)", "Dataset(dataset_x)"],
          x, y_pred.detach(), y_dataset)
data_plot(["Pred(dataset_x)", "Fun(dataset_x)"], x, y_pred.detach(), y_true)


# eval on random data
# %%
model.eval()
x, y_true = data_fun(dataset[0].size(0))
y_pred = model(x)
eval_error = loss_fn(y_pred, y_true).item()
print("eval_random_error: %f" % eval_error)
data_plot(["Pred(rand x)", "Fun(rand x)"], x, y_pred.detach(), y_true)
# %%
