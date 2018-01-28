import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import logging


def conv_concat(x, y, num_cls):
    dim_y = len(y.size())
    bs = y.size(0)

    if dim_y == 1:
        # y = T.extra_ops.to_one_hot(y, self.num_cls)
        # maybe need somewhere to transform to variable
        label = torch.zeros((bs, num_cls))  # zero tensor
        if y.is_cuda:
            label = label.cuda()
        # label = Variable(label) ??
        y = label.scatter_(1, y.data, 1)  # set label to 1 at the indices specified by y --> 1-hot-encoding

    if dim_y == 2:
        # y = y.dimshuffle(0, 1, 'x', 'x')
        y = y.unsqueeze(2).unsqueeze(3)

    assert dim_y == 4, 'Dimension of y != 4'

    # T.concatenate([x, y*T.ones((x.shape[0], y.shape[1], x.shape[2], x.shape[3]))], axis=1)
    y = y * Variable(torch.ones((x.size(0), y.size(1), x.size(2), x.size(3))))
    return torch.cat([x, y], dim=1)


def mlp_concat(x, y, num_cls):
    dim_y = len(y.size())
    bs = y.size(0)

    if dim_y == 1:
        # y = T.extra_ops.to_one_hot(y, self.num_cls)
        # maybe need somewhere to transform to variable
        label = torch.zeros((bs, num_cls))  # zero tensor
        if y.is_cuda:
            label = label.cuda()
        # label = Variable(label) ??
        y = label.scatter_(1, y.data, 1)  # set label to 1 at the indices specified by y --> 1-hot-encoding

    assert dim_y == 2, 'Dimension of y != 2'

    return torch.cat([x, y], dim=1)


def init_weights(model):
    '''initializes the weights of specific NN layers by normal initialization

    Args:
        model (torch.nn.Module): neural net model

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
        if model.bias:
            model.bias.data.fill_(0)
        logger.debug('Weights initialized.')

    elif type(model) in [nn.BatchNorm2d, nn.BatchNorm1d]:   # batch normalizations
        model.weight.data.normal_(mu, sigma)
        model.bias.data.fill_(0)
        logger.debug('Weights initialized.')

    else:
        logger.debug('Initialization failed. Unknown module type: {}'.format(str(type(model))))


class GaussianNoiseLayer(nn.Module):
    def __init__(self, shape, std=0.15):
        super(GaussianNoiseLayer).__init__()
        self.noise = Variable(torch.zeros(shape))  # .cuda()
        self.std = std

    def forward(self, x, deterministic=False):
        if deterministic or (self.std == 0):
            return x
        else:
            return x + self.noise.data.normal_(0, std=self.std)


class MeanOnlyBatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.999):
        super(MeanOnlyBatchNorm, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(num_features))
        # self.reset_parameters()

    def forward(self, x):

        # mu = Variable(torch.mean(input,dim=0, keepdim=True).data, requires_grad=False)
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
