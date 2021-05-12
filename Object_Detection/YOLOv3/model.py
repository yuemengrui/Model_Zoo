# *_*coding:utf-8 *_*
# @Author : yuemengrui
# @Time : 2021-05-12 下午4:57
import torch
import torch.nn as nn


class Residual_Block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)


class Darknet53(nn.Module):

    def __init__(self):
        super(Darknet53, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True)
        )

        self.conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)

        self.residual_block = Residual_Block()

    def forward(self, x):
        return x
