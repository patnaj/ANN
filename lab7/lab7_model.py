# %%
import torch
import os

class RNNModel(torch.nn.Module):

    def __init__(self, in_size, hid_size, layer_size, out_size):
        super(RNNModel, self).__init__()
        self.layer_size = layer_size
        self.hid_size = hid_size
        
        self.block1 = torch.nn.RNN(in_size, hid_size, layer_size, batch_first=True, nonlinearity='relu')
        self.fc = torch.nn.Linear(hid_size, out_size)

    def forward(self, x):
        h0 = torch.autograd.Variable(torch.zeros(self.layer_size, x.size(0), self.hid_size))
        out, hn = self.block1(x, h0)
        # print(out.size(), hn.size())
        out = self.fc(out[:, -1, :]) 
        # print(out.size())
        return out
    
    def save(self, patch='lab7_model.pt'):
        torch.save(self.state_dict(), patch)
        print("Model saved: %s" % patch)

    def load(self, patch='lab7_model.pt'):
        try:
            self.load_state_dict(torch.load(patch))
            print("Model saved: %s" % patch)
        except Exception as e:
            print("Load model error: %s" % e)

    def info(self):
        print(self)
        print("Params: %i" % sum([param.nelement()
                                  for param in self.parameters()]))


# %%
if __name__ == '__main__':
    input = output = 1
    model = RNNModel(2, 12, 4,  1)
    model.info()

    from lab7_data import LoadData
    train, _ = LoadData()
    x, y = next(iter(train))

    print( model(x).size(), y.size())
