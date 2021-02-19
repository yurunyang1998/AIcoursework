import  torch
import  torch.nn as nn
import  torch.nn.functional as F
class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.ConvLayer1 = torch.nn.Conv2d(3,10,5,1,0)
        self.ConvLayer2 = torch.nn.Conv2d(10,16,3,1)
        self.ConvLayer3 = torch.nn.Conv2d(16,10,3,2,1)
        self.fc1 = torch.nn.Linear(10*15*15,128)
        self.fc2 = torch.nn.Linear(128,64)
        self.fc3 = torch.nn.Linear(64,10)

    def forward(self,input):  # input is  3x128x128
        # print(input.size())
        x = self.ConvLayer1(input)    #output shape (10,128,128)
        # print(x.size())
        x = F.max_pool2d(x,(2,2))  # (10,64,64)
        # print(x.size())
        x = self.ConvLayer2(x)   # (16, 64, 64)
        # print(x.size())
        x = F.max_pool2d(x,(2,2)) # (16, 32,32)
        # print(x.size())
        x = self.ConvLayer3(x)  #(10,32,32)
        # print(x.size())
        x = x.view(x.size(0),-1)
        # print(x.size())
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return  x


if __name__ == '__main__':
    net = Module()
    print(net)