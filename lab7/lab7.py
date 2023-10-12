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

# x [100, 6, 2] ->  dane wejsciowe na model
#   100 - batch - ilosc probek pretwarzanych przez model w jednej operacji,     
#   6 - 6 kolejnych pomiarów temperatury,   
#   2 - temeratur oraz miesiąc
print("dane x:", x.size(), x.round().tolist())
# [100, 1] -> dane jakie chcemy uzyskać na wyjsciu modelu dla danych x
#   100 - batch 
#   1 - 7 w koleji pomiar temeratury dla danych x  
print("dane y:", y.size(), y.round().tolist())



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
        
        # normalizacja (tempaeratura w F zamieniamy na skale 0-1, miesiące 12 na skale 0-1)
        inputs = inputs / torch.Tensor([110.,12.])
        labels = labels / torch.Tensor([110.])

        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        outputs = model(inputs) # wynikowa temperatura jest w skali 0-1
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()
        # Model parameters optimization
        optimizer.step()
        
        err += loss.item()
        print("\rerror = %f "%(err), end="")        
    print("\repoch= %d error= %f: \n"%(epoche,err/step))        
model.save()

#test predykcja 
# %%
model.eval()
with torch.no_grad(): # mniejsze urzycie pamieci, wyłączone funcje wstecznej propagacja błedu
    inputs, labels = next(iter(dataset_test))
    #normalizacj
    inputs = inputs / torch.Tensor([110.,12.])
    outputs = model(inputs) * torch.Tensor([110.])
    print("predykowane:", outputs.to(int).tolist())
    print("rzeczywiste:", labels.to(int).tolist())
    print("bład:       ", (outputs-labels).to(int).tolist())
    
    
    #wykres predycki
    idx = torch.arange(-1, labels.size(0)-1, 1/(inputs.size(1)+1))
    tmp = torch.cat([inputs[:,:,0]*torch.Tensor([110.]), labels], dim=1).reshape(-1)
    plt.plot(range(labels.size(0)), outputs, "o-", label="predykcja")
    plt.plot(range(labels.size(0)), labels, "o-", label="pomiar")
    plt.plot( idx, tmp, "go", label="dane wejsciowe x", markersize=2)
    plt.legend()
    plt.title('Test 10 predykcji')
    plt.show()
    
    
    
    
# %%
