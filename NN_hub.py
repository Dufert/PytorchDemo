import torch.nn as nn

class MobileNet(nn.Module):
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
        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)

        return x

class Lenn(nn.Module):
    def __init__(self):
        super(Lenn, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1).cuda(),
            nn.ReLU().cuda(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1).cuda(),
            nn.ReLU().cuda(),
            nn.MaxPool2d(kernel_size=2).cuda(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1).cuda(),
            nn.ReLU().cuda(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1).cuda(),
            nn.ReLU().cuda(),
            nn.MaxPool2d(kernel_size=2).cuda(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1).cuda(),
            nn.ReLU().cuda(),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1).cuda(),
            nn.ReLU().cuda(),
            nn.MaxPool2d(kernel_size=2).cuda(),
        )

        self.out1 = nn.Linear(256 * 3 * 3, 64).cuda()
        self.out2 = nn.Linear(64, 10).cuda()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        x = x.view(x.size(0), -1)
        x = self.out1(x)
        x = self.out2(x)

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
