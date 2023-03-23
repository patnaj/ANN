# %%
import torch
import os


class ConvBlock(torch.nn.Module):
    def __init__(self, in_size, mid_size, out_size = None):
        super(ConvBlock, self).__init__()
        if out_size is None: out_size = mid_size
        self.activation = torch.nn.LeakyReLU()
        self.conv1 = torch.nn.Conv2d(in_size, mid_size, (3, 3), padding=1)
        self.conv2 = torch.nn.Conv2d(mid_size, out_size, (3, 3), padding=1)
        self.maxpool = torch.nn.MaxPool2d(3)
        self.bn = torch.nn.BatchNorm2d(out_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.maxpool(x)
        x = self.bn(x)
        return x


class ConvModel(torch.nn.Module):

    def __init__(self, in_size, mid_size, out_size):
        super(ConvModel, self).__init__()
        self.block1 = ConvBlock(in_size, mid_size)
        self.block2 = ConvBlock(mid_size, out_size)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.activation(x)
        return x

    def save(self, patch='lab4_model.pt'):
        torch.save(self.state_dict(), patch)
        print("Model saved: %s" % patch)

    def load(self, patch='lab4_model.pt'):
        try:
            self.load_state_dict(torch.load(patch))
            print("Model saved: %s" % patch)
        except Exception as e:
            print("Load model error: %s" % e)

    def info(self):
        print(self)
        print("Params: %i" % sum([param.nelement()
                                  for param in model.parameters()]))


# %%
if __name__ == '__main__':
    input = output = 1
    model = ConvModel(input, 2, output)
    model.info()

    # test
    # image set (batch, input[grey], size_x, size_y)
    x = torch.rand(2, input, 100, 100)
    print("x = ", x.size())
    y = model(x)
    print("y = ", y.size())
