import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torchvision as tv
from torch.autograd import Variable
import matplotlib.pyplot as plt

train_data = tv.datasets.MNIST(root="./data_mnist",
                               train=True,
                               transform=tv.transforms.ToTensor(), #会将图片数据0-255变成0-1
                               download=False)
test_data = tv.datasets.MNIST("./data_mnist", train=False, transform=tv.transforms.ToTensor())



train_loader = Data.DataLoader(dataset=train_data, batch_size=1000, shuffle=True, num_workers=0)
test_loader = Data.DataLoader(dataset=test_data, batch_size=500, shuffle=True, num_workers=0)

test_lable = test_data.test_labels

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        def dw_sep(dw_chanenls, dw_stride, sep_outc):

            conv_dw_sep = nn.Sequential(
                nn.Conv2d(in_channels=dw_chanenls, out_channels=dw_chanenls, kernel_size=3, stride=dw_stride, padding=1, groups=dw_chanenls),
                nn.ReLU(),
                nn.Conv2d(in_channels=dw_chanenls, out_channels=sep_outc, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
            )

            return conv_dw_sep

        self.conv1_dw_sep = dw_sep(32, 1, 64)
        self.conv2_dw_sep = dw_sep(64, 2, 128)
        self.conv3_dw_sep = dw_sep(128, 1, 256)
        self.conv4_dw_sep = dw_sep(256, 2, 512)
        self.conv5_dw_sep = dw_sep(512, 1, 512)
        self.conv6_dw_sep = dw_sep(512, 2, 1024)
        self.conv7_dw_sep = dw_sep(1024, 2, 1024)

        self.pool1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=7)
        )
        self.out = nn.Linear(1024*1*1, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_dw_sep(x)
        x = self.conv2_dw_sep(x)
        x = self.conv3_dw_sep(x)
        x = self.conv4_dw_sep(x)
        x = self.conv5_dw_sep(x)
        x = self.conv6_dw_sep(x)
        x = self.conv7_dw_sep(x)
        print(x.shape,x.size(0))

        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        print(x.shape,x.size(0))

        x = self.out(x)

        return x


cnn = VGG16()

xx = np.ones((1, 3, 224, 224), np.float32)
xx = torch.from_numpy(xx)
print(cnn(xx))

# optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
# loss_func = nn.CrossEntropyLoss()
#
# for step, (bx, by) in enumerate(train_loader):
#     bx = Variable(bx)
#     by = Variable(by)
#
#     output = cnn(bx)
#     loss = loss_func(output, by)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     for test_step, (test_x,test_y) in enumerate(test_loader):
#         test_out = cnn(test_x)
#         pre_y = torch.max(test_out, 1)[1].data.numpy().squeeze()
#         accuracy = sum(pre_y == test_y.numpy()) / 500
#         print(accuracy)














