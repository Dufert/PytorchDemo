import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torchvision as tv

from NN_hub import MobileNet
from torch.autograd import Variable
import matplotlib.pyplot as plt

train_data = tv.datasets.MNIST(root="./data_mnist",
                               train=True,
                               transform=tv.transforms.ToTensor(), #会将图片数据0-255变成0-1
                               download=True)
test_data = tv.datasets.MNIST("./data_mnist", train=False, transform=tv.transforms.ToTensor())

train_loader = Data.DataLoader(dataset=train_data, batch_size=1000, shuffle=True, num_workers=0)
test_loader = Data.DataLoader(dataset=test_data, batch_size=500, shuffle=True, num_workers=0)

test_lable = test_data.test_labels

cnn = MobileNet()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()
