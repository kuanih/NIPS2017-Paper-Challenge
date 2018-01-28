import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import weight_norm as wn

from layers import conv_concat, mlp_concat


class Generator(nn.Module):
    def __init__(self, input_size=101, num_classes=10,
                 dense_neurons=8192):

        super(Generator, self).__init__()

        self.num_classes = num_classes

        self.Dense = nn.Linear(input_size, dense_neurons)
        self.Relu = nn.ReLU()
        self.Tanh = nn.Tanh()

        self.Deconv2D_0 = nn.ConvTranspose2d(in_channels=513, out_channels=256,
                                             kernel_size=5, stride=2, padding=2,
                                             output_padding=1, bias=False)
        self.Deconv2D_1 = nn.ConvTranspose2d(in_channels=257, out_channels=128,
                                             kernel_size=5, stride=2, padding=2,
                                             output_padding=1, bias=False)
        self.Deconv2D_2 = wn(nn.ConvTranspose2d(in_channels=129, out_channels=3,
                                                kernel_size=5, stride=2, padding=2,
                                                output_padding=1, bias=False))

        self.BatchNorm1D = nn.BatchNorm1d(dense_neurons)

        self.BatchNorm2D_0 = nn.BatchNorm2d(256)
        self.BatchNorm2D_1 = nn.BatchNorm2d(128)

    def forward(self, y, z):
        x = mlp_concat(y, z, self.num_classes)

        x1 = self.Dense(x)
        x1 = self.Relu(x1)
        x1 = self.BatchNorm1D(x1)

        x2 = x1.resize(z.shape[0], 512, 4, 4)
        x2 = conv_concat(x2, y, self.num_classes)

        x3 = self.Deconv2D_0(x2)                    # output shape (256,8,8) = 8192 * 2
        x3 = self.Relu(x3)
        x3 = self.BatchNorm2D_0(x3)

        x4 = conv_concat(x3, y, self.num_classes)

        x5 = self.Deconv2D_1(x4)                    # output shape (128,16,16) = 8192 * 2 * 2
        x5 = self.Relu(x5)
        x5 = self.BatchNorm2D_1(x5)

        x6 = conv_concat(x5, y, self.num_classes)
        x6 = self.Deconv2D_2(x6)                    # output shape (3, 32, 32) = 3072
        x6 = self.Tanh(x6)

        return x6
