import torch
from torch.autograd import Variable
import numpy as np
from scipy import linalg


class ZCA(object):
    def __init__(self, regularization=1e-5, x=None):
        self.regularization = regularization
        if x is not None:
            self.fit(x)

    def fit(self, x):
        if isinstance(x, np.ndarray):
            s = x.shape
            x = x.copy().reshape((s[0], np.prod(s[1:])))
            m = np.mean(x, axis=0)
            x -= m
            sigma = np.dot(x.T, x) / x.shape[0]
            U, S, V = linalg.svd(sigma)
            tmp = np.dot(U, np.diag(1. / np.sqrt(S + self.regularization)))
            tmp2 = np.dot(U, np.diag(np.sqrt(S + self.regularization)))
            self.ZCA_mat = torch.from_numpy(np.dot(tmp, U.T))
            self.inv_ZCA_mat = torch.from_numpy(np.dot(tmp2, U.T))
            self.mean = m
        else:
            raise NotImplementedError("Init only implemented for np arrays")

    def apply(self, x):

        if isinstance(x, np.ndarray):
            s = x.shape
            return np.dot(x.reshape((s[0], np.prod(s[1:]))) - self.mean, self.ZCA_mat.numpy()).reshape(s)
        elif isinstance(x, torch.Tensor):
            s = x.size()
            dims = len(x.size())
            if dims == 1:
                out = torch.dot(x - self.mean, self.ZCA_mat).view(s)
            else:
                out = torch.mm(x.view(s[0], -1) - torch.from_numpy(self.mean).unsqueeze(0), self.ZCA_mat).view(s)
            return out
        elif isinstance(x, torch.autograd.Variable):
            s = x.size()
            dims = len(x.size())
            subs = Variable(torch.from_numpy(self.mean), requires_grad=False)
            mult = Variable(self.ZCA_mat, requires_grad=False)
            if x.cuda:
                subs, mult = subs.cuda(), mult.cuda()

            if dims == 1:
                out = torch.dot(x - subs, mult).view(s)
            else:
                out = torch.mm(x.view(s[0], -1) - subs.unsqueeze(0), mult).view(s)
            return out
        else:
            raise NotImplementedError("Whitening only implemented for np arrays or torch.Tensors")
