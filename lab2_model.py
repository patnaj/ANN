# %%
import torch
import os


class TinyModel(torch.nn.Module):

    def __init__(self, x=1024, input_range=1):
        super(TinyModel, self).__init__()
        self.activation = torch.nn.LeakyReLU()
        self.linear1 = torch.nn.Linear(input_range, x)
        self.linear2 = torch.nn.Linear(x, x)
        self.linear3 = torch.nn.Linear(x, input_range)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        return x

    def save(self, patch='model.pt'):
        torch.save(self.state_dict(), patch)
        print("Model saved: %s"%patch)

    def load(self, patch='model.pt'):
        try:
            self.load_state_dict(torch.load(patch))
            print("Model saved: %s"%patch)
        except Exception as e:
            print("Error: %s"%e)


    def info(self):
        print(self)
        print("Params: %i" % sum([param.nelement()
                                  for param in model.parameters()]))


if __name__ == '__main__':
    input_range = 2
    model = TinyModel(input_range=input_range)
    model.info()

    # test
    x = torch.rand(10, input_range)
    print("x = ", x.size())
    y = model(x)
    print("y = ", x.size())

# %%

