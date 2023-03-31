import torch

def range_params(i):
    xd = torch.tensor(X)
    x1_min = xd[:,i].min()
    x1_max = xd[:,i].max()
    return torch.arange(x1_min,x1_max,(x1_max-x1_min)/100)

x = range_params(0)
y = range_params(1)

x1= x.unsqueeze(0).expand(y.size(0),-1)
y1= y.unsqueeze(1).expand(-1, x.size(0))
x1.size()
y1.size()
xy = torch.cat([x1.unsqueeze(2), y1.unsqueeze(2)], dim=2)
plt.scatter(xy[:,:,0].view(-1).numpy(), xy[:,:,1].view(-1).numpy())

xy = torch.cat([xy, torch.zeros(xy.size())], dim=2)
