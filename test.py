import torch
import torch.utils.data as Data
import torch.nn as nn
import torchvision as tv
from torch.autograd import Variable
import matplotlib.pyplot as plt

train_data = tv.datasets.MNIST(root="./data_mnist",
                               train=True,
                               transform=tv.transforms.ToTensor(), #会将图片数据0-255变成0-1
                               download=False)
test_data = tv.datasets.MNIST("./data_mnist", train=False, transform=tv.transforms.ToTensor())



train_loader = Data.DataLoader(dataset=train_data, batch_size=1000, shuffle=True, num_workers=0)
test_loader = Data.DataLoader(dataset=test_data, batch_size=200, shuffle=True, num_workers=0)

test_lable = test_data.test_labels

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16,self).__init__()
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
cnn = VGG16()
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














