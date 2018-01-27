import torch
import torch.nn as nn
from torch.autograd import Variable


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
        y = label.scatter(1, y.data, 1)  # set label to 1 at the indices specified by y --> 1-hot-encoding

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
        y = label.scatter(1, y.data, 1)  # set label to 1 at the indices specified by y --> 1-hot-encoding

    assert dim_y == 2, 'Dimension of y != 2'

    return torch.cat([x, y], dim=1)


def init_weights(model, normal=None):
    '''initializes the weights of specific NN layers by xavier normal initialization

    Args:
        model (torch.nn.Module): neural net model
        normal (list): (optional) params list for standard normal initialization, [mu, sigma]

    '''

    if normal:
        mu = normal[0]
        sigma = normal[1]

    if type(model) in [nn.Linear]:  # linear layers
        if normal:
            model.weight.data.normal_(mu, sigma)
            model.bias.data.fill_(0)
        else:
            nn.init.xavier_normal(model.weight.data)
        print('Weights initialized.')
    elif type(model) in [nn.ConvTranspose2d, nn.Conv2d]:  # convolutional layers
        if normal:
            model.weight.data.normal_(mu, sigma)
            model.bias.data.fill_(0)
        else:
            nn.init.xavier_normal(model.weight.data)
        print('Weights initialized.')
    elif type(model) in [nn.BatchNorm2d]:   # init batch normalizations from normal dist
        if normal:
            model.weight.data.normal_(mu, sigma)
            model.bias.data.fill_(0)
        else:
            model.weight.data.normal_(1.0, 0.02)
            model.bias.data.fill_(0)
        print('Weights initialized.')
    else:
        raise NotImplementedError('Initialization failed. Unknown module type: {}'.format(str(type(model))))


