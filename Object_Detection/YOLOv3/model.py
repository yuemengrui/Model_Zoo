# *_*coding:utf-8 *_*
# @Author : yuemengrui
# @Time : 2021-05-12 下午4:57
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.conv(x)


class Residual_Block(nn.Module):

    def __init__(self, in_channels):
        super(Residual_Block, self).__init__()
        self.conv_block = nn.Sequential(
            ConvLayer(in_channels, in_channels // 2, 1, 1, 0),
            ConvLayer(in_channels // 2, in_channels, 3, 1, 1)
        )

    def forward(self, x):
        residual = x
        out = self.conv_block(x)
        out += residual
        return out


class Conv_Block(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels):
        super(Conv_Block, self).__init__()
        self.conv_block = nn.Sequential(
            ConvLayer(in_channels, out_channels, 1, 1, 0),
            ConvLayer(out_channels, mid_channels, 3, 1, 1),
            ConvLayer(mid_channels, out_channels, 1, 1, 0),
            ConvLayer(out_channels, mid_channels, 3, 1, 1),
            ConvLayer(mid_channels, out_channels, 1, 1, 0)
        )

    def forward(self, input):
        return self.conv_block(input)


class Conv_Out(nn.Module):

    def __init__(self, in_channels, out_channels, num_classes):
        super(Conv_Out, self).__init__()
        self.conv_out = nn.Sequential(
            ConvLayer(in_channels, out_channels, 3, 1, 1),
            ConvLayer(out_channels, 3*(num_classes+1+4), 1, 1, 0)
        )

    def forward(self, input):
        return self.conv_out(input)


class UpSampleLayer(nn.Module):
    def __init__(self):
        super(UpSampleLayer, self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode='nearest')


class Darknet53(nn.Module):

    def __init__(self):
        super(Darknet53, self).__init__()
        self.conv_block = ConvLayer(3, 32, 3, 1, 1)

        self.conv1 = ConvLayer(32, 64, 3, 2, 1)

        self.residual_block1 = Residual_Block(64)

        self.conv2 = ConvLayer(64, 128, 3, 2, 1)

        self.residual_block2 = Residual_Block(128)

        self.conv3 = ConvLayer(128, 256, 3, 2, 1)

        self.residual_block3 = Residual_Block(256)

        self.conv4 = ConvLayer(256, 512, 3, 2, 1)

        self.residual_block4 = Residual_Block(512)

        self.conv5 = ConvLayer(512, 1024, 3, 2, 1)

        self.residual_block5 = Residual_Block(1024)

    def forward(self, input):
        x = self.conv_block(input)
        x = self.conv1(x)
        x = self.residual_block1(x)
        x = self.conv2(x)
        for _ in range(2):
            x = self.residual_block2(x)
        x = self.conv3(x)
        for _ in range(8):
            x = self.residual_block3(x)
        feature_52 = x
        print('feature_52: ', feature_52.shape)  # [1, 256, 52, 52]
        x = self.conv4(x)
        for _ in range(8):
            x = self.residual_block4(x)
        feature_26 = x
        print('feature_26: ', feature_26.shape)  # [1, 512, 26, 26]
        x = self.conv5(x)
        for _ in range(4):
            x = self.residual_block5(x)
        feature_13 = x
        print('feature_13', feature_13.shape)  # [1, 1024, 13, 13]

        return feature_52, feature_26, feature_13


class YOLOv3(nn.Module):

    def __init__(self, num_classes=1):
        super(YOLOv3, self).__init__()
        self.backbone = Darknet53()
        self.conv_block_13 = Conv_Block(1024, 512, 1024)
        self.conv_out_13 = Conv_Out(512, 1024, num_classes)
        self.up_26 = nn.Sequential(
            ConvLayer(512, 256, 1, 1, 0),
            UpSampleLayer()
        )

        self.conv_block_26 = Conv_Block(768, 256, 512)
        self.conv_out_26 = Conv_Out(256, 512, num_classes)

        self.up_52 = nn.Sequential(
            ConvLayer(256, 128, 1, 1, 0),
            UpSampleLayer()
        )

        self.conv_block_52 = Conv_Block(384, 128, 256)
        self.conv_out_52 = Conv_Out(128, 256, num_classes)

    def forward(self, input):
        feature_52, feature_26, feature_13 = self.backbone(input)
        conv_13 = self.conv_block_13(feature_13)
        out_13 = self.conv_out_13(conv_13)

        up_26 = self.up_26(conv_13)
        f_26 = torch.cat((up_26, feature_26), dim=1)

        conv_26 = self.conv_block_26(f_26)
        out_26 = self.conv_out_26(conv_26)

        up_52 = self.up_52(conv_26)
        f_52 = torch.cat((up_52, feature_52), dim=1)

        conv_52 = self.conv_block_52(f_52)
        out_52 = self.conv_out_52(conv_52)

        return out_13, out_26, out_52


if __name__ == '__main__':
    model = YOLOv3()
    input = torch.rand((1, 3, 416, 416))

    x = model(input)
    print(x[0].shape)
    print(x[1].shape)
    print(x[2].shape)
