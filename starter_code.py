"""

QUESTION 1

Some helpful code for getting started.


"""
import module
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from imagenet10 import ImageNet10
import  config
import pandas as pd
import os

from config import *

# Gathers the meta data for the images
paths, classes = [], []
for i, dir_ in enumerate(CLASS_LABELS):
    for entry in os.scandir(ROOT_DIR + dir_):
        if (entry.is_file()):
            paths.append(entry.path)
            classes.append(i)

data = {
    'path': paths,
    'class': classes
}

data_df = pd.DataFrame(data, columns=['path', 'class'])
data_df = data_df.sample(frac=1).reset_index(drop=True)  # Shuffles the data

# See what the dataframe now contains
# print("Found", len(data_df), "images.")
# If you want to see the image meta data
# print(data_df.head())



# Split the data into train and test sets and instantiate our new ImageNet10 objects.
train_split = 0.80  # Defines the ratio of train/valid data.

# valid_size = 1.0 - train_size
train_size = int(len(data_df) * train_split)

data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

dataset_train = ImageNet10(
    df=data_df[:train_size],
    transform=data_transform,
)

dataset_valid = ImageNet10(
    df=data_df[train_size:].reset_index(drop=True),
    transform=data_transform,
)

# Data loaders for use during training
train_loader = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=1,
    shuffle=True,
    num_workers=2
)

valid_loader = torch.utils.data.DataLoader(
    dataset_valid,
    batch_size=16,
    shuffle=True,
    num_workers=2
)

# See what you've loaded
# print("len(dataset_train)", len(dataset_train))
# print("len(dataset_valid)", len(dataset_valid))

print("len(train_loader)", len(train_loader))
print("len(valid_loader)", len(valid_loader))

if __name__ == '__main__':



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net = module.Module()
    # net = module.vgg("vgg16")
    net.to(device)
    # net.load_state_dict(torch.load(config.MODELPATH+"modelcheckpoint"))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(net.parameters(), lr=0.1)
    for epoch in range(30):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # input = data[0].to(device)
            # label = data[1].to(device)
            input, label = data[0],data[1]
            # input = input.view(input.size(0), -1)
            output = net(input)
            # label = label.view(1,-1, 1)
            loss = criterion(output,label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss+=loss.item()
            if i % 10 == 1:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
        modelfile = "modelcheckpoint"
        torch.save(net.state_dict(), MODELPATH + modelfile)