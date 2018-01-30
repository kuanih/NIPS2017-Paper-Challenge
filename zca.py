'''
ZCA whitening object for feature de-correlation
'''

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
        '''create object ZCA matrix from input data

        Args:
            x(numpy.ndarray): input data

        '''
        if isinstance(x, np.ndarray):   # check data type

            s = x.shape  # dimensions of s
            x = x.copy().reshape((s[0], np.prod(s[1:])))    # copy data
            m = np.mean(x, axis=0)  # calculate mean
            x -= m  # subtract mean
            sigma = np.dot(x.T, x) / x.shape[0]  # co-variance matrix
            # singular value decomposition (SVD):
            # Factorizes a matrix into two unitary matrices U and V, and a 1-D array S of singular values
            U, S, V = linalg.svd(sigma)
            # calculate regularized principal components
            tmp = np.dot(U, np.diag(1. / np.sqrt(S + self.regularization)))
            # compute final ZCA whitening matrix and convert to pytorch tensor
            self.ZCA_mat = torch.from_numpy(np.dot(tmp, U.T))
            self.mean = m

        else:
            raise NotImplementedError("Init only implemented for np arrays")

    def apply(self, x):
        '''applies ZCA whitening to the input data

        Args:
            x: input data to be whitened, three types are supported:
                numpy.ndarray, torch.Tensor and torch.autograd.Variable

        Returns: transformed data of same type

        '''

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
            subs = Variable(torch.from_numpy(self.mean), requires_grad=True)
            mult = Variable(self.ZCA_mat, requires_grad=True)
            if x.is_cuda:
                subs, mult = subs.cuda(), mult.cuda()

            if dims == 1:
                out = torch.dot(x - subs, mult).view(s)
            else:
                out = torch.mm(x.view(s[0], -1) - subs.unsqueeze(0), mult).view(s)
            return out

        else:
            raise NotImplementedError("Whitening only implemented for ndarrays, Tensors or Variables")
