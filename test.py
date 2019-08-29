import torch
import torch.utils.data as Data
import torch.nn as nn
import torchvision as tv
import matplotlib.pyplot as plt

from torch.autograd import Variable

train_data = tv.datasets.MNIST(root="./data_mnist",
                               train=True,
                               transform=tv.transforms.ToTensor(), #会将图片数据0-255变成0-1
                               download=True)
test_data = tv.datasets.MNIST("./data_mnist", train=False, transform=tv.transforms.ToTensor())

train_loader = Data.DataLoader(dataset=train_data, batch_size=1000, shuffle=True, num_workers=0)
test_loader = Data.DataLoader(dataset=test_data, batch_size=200, shuffle=True, num_workers=0)

test_lable = test_data.test_labels


class Lenn(nn.Module):
    def __init__(self):
        super(Lenn,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.out = nn.Linear(32*7*7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)

        return x


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(

            # 1-1 conv layer
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            # 1-2 conv layer
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            # 1 Pooling layer
            nn.MaxPool3d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(

            # 2-1 conv layer
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),

            # 2-2 conv layer
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),

            # 2 Pooling lyaer
            nn.MaxPool3d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(

            # 3-1 conv layer
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),

            # 3-2 conv layer
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),

            # 3-3 conv layer
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),

            # 3 Pooling layer
            nn.MaxPool3d(kernel_size=2, stride=2))

        self.layer4 = nn.Sequential(

            # 4-1 conv layer
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),

            # 4-2 conv layer
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),

            # 4-3 conv layer
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),

            # 4 Pooling layer
            nn.MaxPool3d(kernel_size=2, stride=2))

        self.layer5 = nn.Sequential(

            # 5-1 conv layer
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),

            # 5-2 conv layer
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),

            # 5-3 conv layer
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),

            # 5 Pooling layer
            nn.MaxPool3d(kernel_size=2, stride=2))

        self.layer6 = nn.Sequential(

            # 6 Fully connected layer
            # Dropout layer omitted since batch normalization is used.
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU())

        self.layer7 = nn.Sequential(

            # 7 Fully connected layer
            # Dropout layer omitted since batch normalization is used.
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU())

        self.layer8 = nn.Sequential(

            # 8 output layer
            nn.Linear(4096, 1000),
            nn.BatchNorm1d(1000),
            nn.Softmax())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        vgg16_features = out.view(out.size(0), -1)
        out = self.layer6(vgg16_features)
        out = self.layer7(out)
        out = self.layer8(out)

        return vgg16_features, out


cnn = Lenn()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()

for step, (bx, by) in enumerate(train_loader):
    bx = Variable(bx)
    by = Variable(by)

    output = cnn(bx)
    loss = loss_func(output, by)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    for test_step, (test_x,test_y) in enumerate(test_loader):
        test_out = cnn(test_x)
        pre_y = torch.max(test_out, 1)[1].data.numpy().squeeze()
        accuracy = sum(pre_y == test_y.numpy()) / 200
        print(accuracy)














