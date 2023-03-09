# init block
#%%
import torch
from lab2_model import TinyModel as Model
model = Model()

#data generator
def data_gen(batch=30, range = 2*torch.pi):
    x = torch.rand(batch,1)*range
    return x, torch.sin(x)

#test
x, y = data_gen(5)
print("x=",x)
print("y=",y)

#train block
#%%
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss(reduction="mean")
for epoche in range(30):
    err = 0
    for step in range(50):
        # Every data instance is an input + label pair
        inputs, labels = data_gen(100)
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
    print("\repoch= %d error= %f: \n"%(epoche,err))        


#prediction/test block        
#%%
model.eval()
x, y_true = data_gen(10)
y_pred = model(x)
print("     x=",x)
print("y_pred=",y_pred)
print("y_true=",y_true)
print("L1_Err=",y_true-y_pred)
