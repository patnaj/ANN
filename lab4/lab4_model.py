# %%
import torch
import os


class ConvModel(torch.nn.Module):

    def __init__(self, x=2, in_size=1, out_size=1):
        super(ConvModel, self).__init__()
        self.activation2 = torch.nn.Sigmoid()
        # self.activation2 = torch.nn.ReLU()
        self.activation = torch.nn.LeakyReLU()
        # self.pad = torch.nn.ReflectionPad2d(1)
        self.conv1 = torch.nn.Conv2d(in_size, x, (3,3), padding=1)
        # self.conv1 = torch.nn.Conv2d(in_size, x, (3,3), padding=0)
        self.conv2 = torch.nn.Conv2d(x, x, (1,1))
        self.conv3 = torch.nn.Conv2d(x, out_size, (1,1))

    def forward(self, x):
        # x = self.pad(x)
        # x = torch.nn.functional.pad(x, pad=(1, 1, 1, 1), mode='replicate')

        x = self.conv1(x)
        x = self.activation(x)        
        x = self.conv2(x)
        x = self.activation(x)        
        x = self.conv3(x)
        x = self.activation2(x)
        return x

    def save(self, patch='lab4_model.pt'):
        torch.save(self.state_dict(), patch)
        print("Model saved: %s"%patch)

    def load(self, patch='lab4_model.pt'):
        try:
            self.load_state_dict(torch.load(patch))
            print("Model loaded: %s"%patch)
        except Exception as e:
            print("Load model error: %s"%e)


    def info(self):
        print(self)
        print("Params: %i" % sum([param.nelement()
                                  for param in self.parameters()]))


if __name__ == '__main__':
    input = output = 1
    model = ConvModel(2, input, output)
    model.info()

    # test 
    # image set (batch, input[grey], size_x, size_y)
    x = torch.rand(2, input, 10, 10)
    print("x = ", x.size())
    y = model(x)
    print("y = ", x.size())

# %%

