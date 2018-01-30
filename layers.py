'''
custom network layers and dependent functions
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import logging


def conv_concat(x, y, num_cls):
    '''concatenates output from a convolutional layer with a target vector

    Args:
        x(torch.autograd.Variable): output from convolutional layer
        y(torch.autograd.Variable): multiclass target vector or matrix
        num_cls(int): number of target classes

    Returns: concatenated tensor variable

    '''
    dim_y = len(y.size())
    bs = y.size(0)
    y = y.long()

    if dim_y == 1:  # in case of vector: 1-hot-encode whole batch

        label = Variable(torch.zeros((bs, num_cls)))  # zero tensor
        if y.is_cuda:
            label = label.cuda()
        y = label.scatter_(1, y.unsqueeze(1), 1)  # set label to 1 at the indices specified by y --> 1-hot-encoding
        dim_y = len(y.size())   # update dimension

    if dim_y == 2:  # in case of matrix: increase dimension
        y = y.unsqueeze(2).unsqueeze(3)
        dim_y = len(y.size())  # update dimension

    assert dim_y == 4, 'Dimension of y != 4'

    # reformat target
    factor = Variable(torch.ones((x.size(0), y.size(1), x.size(2), x.size(3))))
    if x.is_cuda:
        factor = factor.cuda()
    y = y * factor

    # concatenate and return
    return torch.cat([x, y], dim=1)


def mlp_concat(x, y, num_cls):
    '''concatenates output from a linear (MLP) layer with a target vector

    Args:
        x(torch.autograd.Variable): output from linear layer
        y(torch.autograd.Variable): multiclass target
        num_cls(int): number of target classes

    Returns: concatenated tensor variable

    '''
    dim_y = len(y.size())
    bs = y.size(0)
    y = y.long()

    if dim_y == 1:  # in case of vector: 1-hot-encode whole batch
        label = Variable(torch.zeros((bs, num_cls)))  # zero tensor
        if y.is_cuda:
            label = label.cuda()
        y = label.scatter_(1, y.unsqueeze(1), 1)  # set label to 1 at the indices specified by y --> 1-hot-encoding
        dim_y = len(y.size())  # update dimension

    assert dim_y == 2, 'Dimension of y != 2'

    # type cast for concatination
    typ = x.data.type()
    y = y.type(typ)

    return torch.cat([x, y], dim=1)


def init_weights(model):
    '''initializes the weights of specific NN layers by normal initialization

    Args:
        model (torch.nn.Module): neural net model layers

    '''
    logger = logging.getLogger(__name__)

    # initialization params:
    mu = 1.0
    sigma = 0.05

    if type(model) in [nn.Linear]:  # linear layers
        model.weight.data.normal_(mu, sigma)
        model.bias.data.fill_(0)
        logger.debug('Weights initialized.')

    elif type(model) in [nn.ConvTranspose2d, nn.Conv2d]:  # convolutional layers
        model.weight.data.normal_(mu, sigma)
        if model.bias is not None:
            model.bias.data.fill_(0)
        logger.debug('Weights initialized.')

    elif type(model) in [nn.BatchNorm2d, nn.BatchNorm1d]:   # batch normalizations
        model.weight.data.normal_(mu, sigma)
        model.bias.data.fill_(0)
        logger.debug('Weights initialized.')

    else:
        logger.debug('Initialization failed. Unknown module type: {}'.format(str(type(model))))


class Gaussian_NoiseLayer(nn.Module):
    def __init__(self, std=0.15):
        super(Gaussian_NoiseLayer, self).__init__()
        self.std = std

    def forward(self, x, cuda, deterministic=False):
        if deterministic or (self.std == 0):
            return x
        else:
            if cuda:
                return x + Variable(torch.randn(x.size()) * self.std).cuda()
            else:
                return x + Variable(torch.randn(x.size()) * self.std)

class MeanOnlyBatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.999):
        super(MeanOnlyBatchNorm, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(num_features))

    def forward(self, x):
        if self.training is True:
            mu = x.mean(dim=0, keepdim=True)
            mu = self.momentum * mu + (1 - self.momentum) * Variable(self.running_mean)
            self.running_mean = mu.data
            return x.add_(-mu)
        else:
            return x.add_(-Variable(self.running_mean))



def rampup(epoch):
    if epoch < 80:
        p = max(0.0, float(epoch)) / float(80)
        p = 1.0 - p
        return math.exp(-p*p*5.0)
    else:
        return 1.0


def rampdown(epoch):
    if epoch >= (300 - 50):
        ep = (epoch - (300 - 50)) * 0.5
        return math.exp(-(ep * ep) / 50)
    else:
        return 1.0
