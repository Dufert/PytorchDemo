import torch
import torch.utils.data as Data
import torch.nn as nn
import torchvision as tv

from NN_hub import Lenn, VGG16, MobileNet

from torch.autograd import Variable

train_data = tv.datasets.MNIST(root="./data_mnist",
                               train=True,
                               transform=tv.transforms.ToTensor(), #会将图片数据0-255变成0-1
                               download=True)
test_data = tv.datasets.MNIST("./data_mnist", train=False, transform=tv.transforms.ToTensor())

train_loader = Data.DataLoader(dataset=train_data, batch_size=100, shuffle=True, num_workers=0)
test_loader = Data.DataLoader(dataset=test_data, batch_size=5000, shuffle=True, num_workers=0)

test_lable = test_data.test_labels

cnn = Lenn()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()

cuda_gpu = torch.cuda.is_available()

for train_step, (train_x, train_y) in enumerate(train_loader):
    if cuda_gpu:
        train_x = Variable(train_x).cuda()
        train_y = Variable(train_y).cuda()
    else:
        train_x = Variable(train_x)
        train_y = Variable(train_y)

    output = cnn(train_x)
    loss = loss_func(output, train_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_out = output.cpu()
    tra_y = torch.max(train_out, 1)[1].data.numpy().squeeze()
    train_accuracy = sum(tra_y == train_y.cpu().numpy()) / 100
    print('train_acc',train_accuracy)


for test_step, (test_x,test_y) in enumerate(test_loader):
    if cuda_gpu:
        test_x = Variable(test_x).cuda()
        test_y = Variable(test_y).cuda()
    else:
        test_x = Variable(test_x)
        test_y = Variable(test_y)

    test_out = cnn(test_x).cpu()
    pre_y = torch.max(test_out, 1)[1].data.numpy().squeeze()
    accuracy = sum(pre_y == test_y.cpu().numpy()) / 5000
    print('test_acc',accuracy)


torch.save(cnn.state_dict(), 'params.pkl')











