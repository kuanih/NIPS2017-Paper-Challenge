'''
NN Models for SGAN
'''

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm as wn

from layers import conv_concat, mlp_concat, init_weights, Gaussian_NoiseLayer, MeanOnlyBatchNorm



### MODEL STRUCTURES ###

# generator y2x: p_g(x, y) = p(y) p_g(x | y) where x = G(z, y), z follows p_g(z)
class Generator(nn.Module):
    def __init__(self, input_size, num_classes, dense_neurons, weight_init=True):
        super(Generator, self).__init__()

        self.logger = logging.getLogger(__name__)  # initialize logger

        self.num_classes = num_classes

        self.Dense = nn.Linear(input_size, dense_neurons)
        self.Relu = nn.ReLU()
        self.Tanh = nn.Tanh()

        self.Deconv2D_0 = nn.ConvTranspose2d(in_channels=522, out_channels=256,
                                             kernel_size=5, stride=2, padding=2,
                                             output_padding=1, bias=False)
        self.Deconv2D_1 = nn.ConvTranspose2d(in_channels=266, out_channels=128,
                                             kernel_size=5, stride=2, padding=2,
                                             output_padding=1, bias=False)
        self.Deconv2D_2 = wn(nn.ConvTranspose2d(in_channels=138, out_channels=3,
                                                kernel_size=5, stride=2, padding=2,
                                                output_padding=1, bias=False))

        self.BatchNorm1D = nn.BatchNorm1d(dense_neurons)

        self.BatchNorm2D_0 = nn.BatchNorm2d(256)
        self.BatchNorm2D_1 = nn.BatchNorm2d(128)

        if weight_init:
            # initialize weights for all conv and lin layers
            self.apply(init_weights)
            # log network structure
            self.logger.debug(self)

    def forward(self, z, y):
        x = mlp_concat(z, y, self.num_classes)

        x = self.Dense(x)
        x = self.Relu(x)
        x = self.BatchNorm1D(x)

        x = x.resize(z.size(0), 512, 4, 4)
        x = conv_concat(x, y, self.num_classes)

        x = self.Deconv2D_0(x)                    # output shape (256,8,8) = 8192 * 2
        x = self.Relu(x)
        x = self.BatchNorm2D_0(x)

        x = conv_concat(x, y, self.num_classes)

        x = self.Deconv2D_1(x)                    # output shape (128,16,16) = 8192 * 2 * 2
        x = self.Relu(x)
        x = self.BatchNorm2D_1(x)

        x = conv_concat(x, y, self.num_classes)
        x = self.Deconv2D_2(x)                    # output shape (3, 32, 32) = 3072
        x = self.Tanh(x)

        return x


# classifier module
class ClassifierNet(nn.Module):
    def __init__(self, in_channels, weight_init=True):
        super(ClassifierNet, self).__init__()

        self.logger = logging.getLogger(__name__)  # initialize logger

        self.gaussian = Gaussian_NoiseLayer()

        self.conv1a = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3,
                                stride=1, padding=1)
        self.convWN1 = MeanOnlyBatchNorm([1, 128, 32, 32])
        self.conv1b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,
                                stride=1, padding=1)
        self.conv_relu = nn.LeakyReLU(negative_slope=0.1)
        self.conv1c = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,
                                stride=1, padding=1)
        self.conv_maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout2d(p=0.5)
        self.conv2a = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                                stride=1, padding=1)
        self.convWN2 = MeanOnlyBatchNorm([1, 256, 16, 16])
        self.conv2b = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                                stride=1, padding=1)
        self.conv2c = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                                stride=1, padding=1)
        self.conv_maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout2d(p=0.5)
        self.conv3a = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,
                                stride=1, padding=0)  # output[6,6]
        self.convWN3a = MeanOnlyBatchNorm([1, 512, 6, 6])
        self.conv3b = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1,
                                stride=1, padding=0)
        self.convWN3b = MeanOnlyBatchNorm([1, 256, 6, 6])
        self.conv3c = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1,
                                stride=1, padding=0)
        self.convWN3c = MeanOnlyBatchNorm([1, 128, 6, 6])

        self.conv_globalpool = nn.AdaptiveAvgPool2d(6)

        self.dense = nn.Linear(in_features=128 * 6 * 6, out_features=10)
        self.smx = nn.Softmax()
        #self.WNfinal = MeanOnlyBatchNorm([1, 128, 6, 6])

        if weight_init:
            # initialize weights for all conv and lin layers
            self.apply(init_weights)
            # log network structure
            self.logger.debug(self)

    def forward(self, x, cuda):
        x = self.gaussian(x, cuda=cuda)
        x = self.convWN1(self.conv_relu(self.conv1a(x)))
        x = self.convWN1(self.conv_relu(self.conv1b(x)))
        x = self.convWN1(self.conv_relu(self.conv1c(x)))
        x = self.conv_maxpool1(x)
        x = self.dropout1(x)
        x = self.convWN2(self.conv_relu(self.conv2a(x)))
        x = self.convWN2(self.conv_relu(self.conv2b(x)))
        x = self.convWN2(self.conv_relu(self.conv2c(x)))
        x = self.conv_maxpool2(x)
        x = self.dropout2(x)
        x = self.convWN3a(self.conv_relu(self.conv3a(x)))
        x = self.convWN3b(self.conv_relu(self.conv3b(x)))
        x = self.convWN3c(self.conv_relu(self.conv3c(x)))
        x = self.conv_globalpool(x)
        x = x.view(-1, 128 * 6 * 6)
        #x = self.WNfinal(self.smx(self.dense(x)))
        x = self.smx(self.dense(x))
        return x


# inference module
class InferenceNet(nn.Module):
    def __init__(self, in_channels, n_z, weight_init=True):
        super(InferenceNet, self).__init__()

        self.logger = logging.getLogger(__name__)  # initialize logger

        self.inf02 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4,
                               stride=2, padding=1)
        self.inf03 = nn.BatchNorm2d(64)
        self.inf11 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4,
                               stride=2, padding=1)
        self.inf12 = nn.BatchNorm2d(128)
        self.inf21 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,
                               stride=2, padding=1)
        self.inf22 = nn.BatchNorm2d(256)
        self.inf31 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,
                               stride=2, padding=1)
        self.inf32 = nn.BatchNorm2d(512)
        self.inf4 = nn.Linear(in_features=512*2*2, out_features=n_z)

        if weight_init:
            # initialize weights for all conv and lin layers
            self.apply(init_weights)
            # log network structure
            self.logger.debug(self)

    def forward(self, x):
        x = F.leaky_relu(self.inf03(self.inf02(x)))
        x = F.leaky_relu(self.inf12(self.inf11(x)))
        x = F.leaky_relu(self.inf22(self.inf21(x)))
        x = F.leaky_relu(self.inf32(self.inf31(x)))
        x = x.view(-1, 512*2*2)
        x = self.inf4(x)

        return x


# discriminator xy2p: test a pair of input comes from p(x, y) instead of p_c or p_g
class DConvNet1(nn.Module):
    '''
    1st convolutional discriminative net (discriminator xy2p)
    --> does a pair of input come from p(x, y) instead of p_c or p_g ?
    '''

    def __init__(self, channel_in, num_classes, p_dropout=0.2, weight_init=True):
        super(DConvNet1, self).__init__()

        self.logger = logging.getLogger(__name__)  # initialize logger

        self.num_classes = num_classes

        # general reusable layers:
        self.LReLU = nn.LeakyReLU(negative_slope=0.2)  # leaky ReLU activation function
        self.sgmd = nn.Sigmoid()  # sigmoid activation function
        self.drop = nn.Dropout(p=p_dropout)  # dropout layer

        # input -->
        # drop
        # ConvConcat

        self.conv1 = wn(nn.Conv2d(in_channels=channel_in, out_channels=32,
                                  kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False))
        # LReLU
        # ConvConcat

        self.conv2 = wn(nn.Conv2d(in_channels=32, out_channels=32,
                                  kernel_size=(3, 3), stride=2, padding=1, bias=False))
        # LReLU
        # drop
        # ConvConcat

        self.conv3 = wn(nn.Conv2d(in_channels=32, out_channels=64,
                                  kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False))
        # LReLU
        # ConvConcat

        self.conv4 = wn(nn.Conv2d(in_channels=64, out_channels=64,
                                  kernel_size=(3, 3), stride=2, padding=1, bias=False))
        # LReLU
        # drop
        # ConvConcat

        self.conv5 = wn(nn.Conv2d(in_channels=64, out_channels=128,
                                  kernel_size=(3, 3), stride=(1, 1), padding=0, bias=False))
        # LReLU
        # ConvConcat

        self.conv6 = wn(nn.Conv2d(in_channels=128, out_channels=128,
                                  kernel_size=(3, 3), stride=(1, 1), padding=0, bias=False))
        # LReLU

        self.globalPool = nn.AdaptiveAvgPool1d(output_size=128)

        # MLPConcat

        self.lin = nn.Linear(in_features=128,
                             out_features=1)
        # smg

        if weight_init:
            # initialize weights for all conv and lin layers
            self.apply(init_weights)
            # log network structure
            self.logger.debug(self)

    def forward(self, x, y):
        # x: (bs, channel_in, dim_input)
        # y: (bs, 1)

        x0 = self.drop(x)
        x0 = self.conv_concat(x0, y, self.num_classes)

        x1 = self.LReLU(self.conv1(x0))
        x1 = self.conv_concat(x1, y, self.num_classes)

        x2 = self.LReLU(self.conv2(x1))
        x2 = self.drop(x2)
        x2 = self.conv_concat(x2, y, self.num_classes)

        x3 = self.LReLU(self.conv3(x2))
        x3 = self.conv_concat(x3, y, self.num_classes)

        x4 = self.LReLU(self.conv4(x3))
        x4 = self.drop(x4)
        x4 = self.conv_concat(x4, y, self.num_classes)

        x5 = self.LReLU(self.conv5(x4))
        x5 = self.conv_concat(x5, y, self.num_classes)

        x6 = self.LReLU(self.conv6(x5))

        x_pool = self.globalPool(x6)
        x_out = self.mlp_concat(x_pool, y)

        out = self.sgmd(self.lin(x_out))

        return out


# discriminator xz
class DConvNet2(nn.Module):
    '''
    2nd convolutional discriminative net (discriminator xz)
    '''

    def __init__(self, n_z, channel_in, num_classes, weight_init=True):
        super(DConvNet2, self).__init__()

        self.logger = logging.getLogger(__name__)  # initialize logger

        self.num_classes = num_classes

        # general reusable layers:
        self.LReLU = nn.LeakyReLU(negative_slope=0.2)  # leaky ReLU activation function
        self.sgmd = nn.Sigmoid()  # sigmoid activation function

        # z input -->
        self.lin_z0 = nn.Linear(in_features=n_z,
                                out_features=512)
        # LReLU

        self.lin_z1 = nn.Linear(in_features=512,
                                out_features=512)
        # LReLU

        # -------------------------------------

        # x input -->
        self.conv_x0 = nn.Conv2d(in_channels=channel_in, out_channels=128,
                                 kernel_size=(5, 5), stride=2, padding=2, bias=False)
        # LReLU

        self.conv_x1 = nn.Conv2d(in_channels=128, out_channels=256,
                                 kernel_size=(5, 5), stride=2, padding=2, bias=False)
        # LReLU
        self.bn1 = nn.BatchNorm2d(num_features=256)

        self.conv_x2 = nn.Conv2d(in_channels=256, out_channels=512,
                                 kernel_size=(5, 5), stride=2, padding=2, bias=False)
        # LReLU
        self.bn2 = nn.BatchNorm2d(num_features=512)

        # -------------------------------------

        # concat x & z -->
        self.lin_f0 = nn.Linear(in_features=8704,
                                out_features=1024)
        # LReLU

        self.lin_f1 = nn.Linear(in_features=1024,
                                out_features=1)
        # smg

        if weight_init:
            # initialize weights for all conv and lin layers
            self.apply(init_weights)
            # log network structure
            self.logger.debug(self)

    def forward(self, z, x):
        # x: (bs, channel_in, dim_input)
        # z: (bs, n_z)

        z0 = self.LReLU(self.lin_z0(z))
        z_out = self.LReLU(self.lin_z1(z0))

        x0 = self.LReLU(self.conv_x0(x))
        x1 = self.LReLU(self.conv_x1(x0))
        x1 = self.bn1(x1)
        x_out = self.LReLU(self.conv_x2(x1))
        x_out = self.bn2(x_out)

        dims = x_out.size()
        fusion = torch.cat([x_out.view(dims[0], -1).squeeze(-1).squeeze(-1), z_out], dim=1)

        f_out = self.LReLU(self.lin_f0(fusion))
        out = self.sgmd(self.lin_f1(f_out))

        return out
