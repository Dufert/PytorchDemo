import torch
import torch.utils.data as Data
import torch.nn as nn
import torchvision as tv

from NN_hub import Lenn, VGG16, MobileNet
from mDataSet import mDataset
from torch.autograd import Variable

learning_rate = 0.001
train_num = 10
train_batch_size = 5
test_batch_size = 40

mtrainDataSet = mDataset("g:/Library/Hub_PedestrianDetectionDataSets/INRIAPerson/inriaperson_cut_data/train/")
mtestDataSet = mDataset("g:/Library/Hub_PedestrianDetectionDataSets/INRIAPerson/inriaperson_cut_data/test/")

train_loader = Data.DataLoader(mtrainDataSet, batch_size=train_batch_size, shuffle=True, num_workers=0)
test_loader = Data.DataLoader(mtestDataSet, batch_size=test_batch_size, shuffle=True, num_workers=0)

# train_data = tv.datasets.MNIST(root="./data_mnist",
#                                train=True,
#                                transform=tv.transforms.ToTensor(), #会将图片数据0-255变成0-1
#                                download=True)
# test_data = tv.datasets.MNIST("./data_mnist", train=False, transform=tv.transforms.ToTensor())

# train_loader = Data.DataLoader(dataset=train_data, batch_size=100, shuffle=True, num_workers=0)
# test_loader = Data.DataLoader(dataset=test_data, batch_size=5000, shuffle=True, num_workers=0)

cnn = MobileNet()
optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss()

cuda_gpu = torch.cuda.is_available()

for train_now_num in range(train_num):
    train_all_accuracy = 0
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
        train_accuracy = sum(tra_y == train_y.cpu().numpy()) / train_batch_size
        train_all_accuracy = train_all_accuracy + train_accuracy

    print('train_acc:', train_all_accuracy/(train_step + 1))

    test_all_accuracy = 0
    for test_step, (test_x, test_y) in enumerate(test_loader):
        if cuda_gpu:
            test_x = Variable(test_x).cuda()
            test_y = Variable(test_y).cuda()
        else:
            test_x = Variable(test_x)
            test_y = Variable(test_y)

        test_out = cnn(test_x).cpu()
        pre_y = torch.max(test_out, 1)[1].data.numpy().squeeze()
        accuracy = sum(pre_y == test_y.cpu().numpy()) / test_batch_size
        test_all_accuracy = test_all_accuracy + accuracy

    print('test_acc:', test_all_accuracy/(test_step + 1))

# torch.save(cnn.state_dict(), 'params.pkl')



