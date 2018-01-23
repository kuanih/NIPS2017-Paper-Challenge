import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


def read_data():

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    return trainloader


batch_size = 1


class Generator(nn.Module):
    def __init__(self, input_size=101, dense_neurons=8192,
                 num_output_features=3072):

        super(Generator, self).__init__()

        self.Dense = nn.Linear(input_size, dense_neurons)
        self.Relu = nn.ReLU()
        self.Tanh = nn.Tanh()

        self.Deconv2D_0 = nn.ConvTranspose2d(in_channels=512, out_channels=256,
                                             kernel_size=(5, 5), bias=False)
        self.Deconv2D_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128,
                                             kernel_size=(5, 5), bias=False)
        self.Deconv2D_2 = nn.ConvTranspose2d(in_channels=128, out_channels=3,
                                             kernel_size=(5, 5), bias=False)

        self.BatchNorm1D = nn.BatchNorm1d(dense_neurons)

        self.BatchNorm2D_0 = nn.BatchNorm2d(dense_neurons * 2)
        self.BatchNorm2D_1 = nn.BatchNorm2d(dense_neurons * 4)
        self.BatchNorm2D_2 = nn.BatchNorm2d(num_output_features)

    def mlp_concat(self, z, y):
        return z + y

    def conv_concat_layer(self, x, y):
        return x + y

    def weigth_norm(self, x):
        return x

    def forward(self, z, y):
        x = self.mlp_concat(z, y)

        x1 = self.Dense(x)
        x1 = self.Relu(x1)
        x1 = self.BatchNorm1D(x1)

        x2 = x1.resize_(-1, 512, 4, 4)
        x2 = self.conv_concat_layer(x2, y)

        x3 = self.Deconv2D_0(x2)                # output shape (256,8,8) = 8192 * 2
        x3 = self.Relu(x3)
        x3 = self.BatchNorm2D_0(x3)

        x4 = self.conv_concat_layer(x3, y)

        x5 = self.Deconv2D_1(x4)                # output shape (128,16,16) = 8192 * 2 * 2
        x5 = self.Relu(x5)
        x5 = self.BatchNorm2D_1(x5)

        x6 = self.conv_concat_layer(x5, y)
        x6 = self.Deconv2D_2(x6)                # output shape (3, 32, 32) = 3072
        x6 = self.Tanh(x6)

        x7 = self.weigth_norm(x6)

        return x7
