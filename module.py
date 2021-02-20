import  torch
import  torch.nn as nn
import  torch.nn.functional as F
# class Module(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # self.ConvLayer1 = torch.nn.Conv2d(3,10,5,1,0)
#         # self.ConvLayer2 = torch.nn.Conv2d(10,16,3,1)
#         # self.ConvLayer3 = torch.nn.Conv2d(16,10,3,2,1)
#         self.Layer1 = nn.Sequential(
#                 torch.nn.Linear(3*128*128,10),
#                 nn.Sigmoid())
#         # self.Layer2 = nn.Sequential(
#         #         torch.nn.Linear(3*64*64, 10),
#         #         # nn.ReLU(True)
#         # )
#         # self.Layer3 = nn.Sequential(
#         #     torch.nn.Linear(3*32*32, 10)
#         # )
#         # self.fc1 = torch.nn.Linear(10*15*15,128)
#         # self.fc2 = torch.nn.Linear(128,64)
#         # self.fc3 = torch.nn.Linear(64,10)
#
#     def forward(self,inputs):
#         # print(inputs.size())
#         # x = inputs.view(1,-1)
#         x = self.Layer1(inputs)
#         # x = self.Layer2(x)
#         # x = self.Layer3(x)
#
#         # x = self.fc1(F.sigmoid(x))
#         # x = self.fc2(F.sigmoid(x))
#         # x = self.fc3(F.sigmoid(x))
#         return x
#     # def forward(self,input):  # input is  3x128x128
#     #     # print(input.size())
#     #     x = self.ConvLayer1(input)    #output shape (10,128,128)
#     #     x = torch.nn.functional.sigmoid(x)
#     #     # print(x.size())
#     #     x = F.max_pool2d(x,(2,2))  # (10,64,64)
#     #     x = torch.nn.functional.sigmoid(x)
#     #
#     #     # print(x.size())
#     #     x = self.ConvLayer2(x)   # (16, 64, 64)
#     #     x = torch.nn.functional.sigmoid(x)
#     #
#     #     # print(x.size())
#     #     x = F.max_pool2d(x,(2,2)) # (16, 32,32)
#     #     x = torch.nn.functional.sigmoid(x)
#     #
#     #     # print(x.size())
#     #     x = self.ConvLayer3(x)  #(10,32,32)
#     #     x = torch.nn.functional.sigmoid(x)
#     #
#     #     # print(x.size())
#     #     x = x.view(x.size(0),-1)
#     #     # print(x.size())
#     #     x = self.fc1(x)
#     #     x = torch.nn.functional.sigmoid(x)
#     #
#     #     x = self.fc2(x)
#     #     x = torch.nn.functional.relu(x)
#     #     x = self.fc3(x)
#     #     return  x
#

class Module(nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 29 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.size())
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.size())
        x = x.view(-1, 16 * 29 * 29)
        # print(x.size())
        x = F.relu(self.fc1(x))
        # print(x.size())
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class VGG(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        #先会进行一个展平处理
        self.classifier = nn.Sequential( #定义分类网络结构
            nn.Dropout(p=0.5), #减少过拟合
            nn.Linear(512*7*7, 2048),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)#展平处理
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self): #初始化权重函数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)#初始化偏置为0
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_features(cfg: list):#提取特征函数
    layers = [] #存放创建每一层结构
    in_channels = 3 #RGB
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)   #通过非关键字参数传入


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], #数字为卷积层个数，M为池化层结构
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg(model_name="vgg16", **kwargs): #**字典
    try:
        cfg = cfgs[model_name]
    except:
        print("Warning: model number {} not in cfgs dict!".format(model_name))
        exit(-1)
    model = VGG(make_features(cfg), **kwargs) #
    return model



if __name__ == '__main__':
    net = Module()
    print(net)