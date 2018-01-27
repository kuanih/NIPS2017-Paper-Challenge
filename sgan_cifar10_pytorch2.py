import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
import numpy as np
import time
import math
import utils
from layers import conv_concat, mlp_concat, init_weights
from sklearn.metrics import accuracy_score
import os
import logging
from torch.nn.utils import weight_norm as wn


### GLOBAL PARAMS ###
# --------------------------------------------------------------
BATCH_SIZE = 200
BATCH_SIZE_EVAL = 200
NUM_CLASSES = 10
NUM_LABELLED = 4000
SSL_SEED = 1
NP_SEED = 1234

# data dependent
IN_CHANNELS = 3

# evaluation
VIS_EPOCH = 1
EVAL_EPOCH = 1

# C
SCALED_UNSUP_WEIGHT_MAX = 100.0
# G
N_Z = 100

# optimization
B1 = 0.5  # moment1 in Adam
LR = 3e-4
LR_CLA = 3e-3
NUM_EPOCHS = 1000
NUM_EPOCHS_PRE = 20
ANNEAL_EPOCH = 200
ANNEAL_EVERY_EPOCH = 1
ANNEAL_FACTOR = 0.995
ANNEAL_FACTOR_CLA = 0.99

path_out = "./results"



'''
data
'''
train_x, train_y = utils.load('./cifar10/', 'train')
eval_x, eval_y = utils.load('./cifar10/', 'test')

train_y = np.int32(train_y)
eval_y = np.int32(eval_y)
x_unlabelled = train_x.copy()

rng_data = np.random.RandomState(SSL_SEED)
inds = rng_data.permutation(train_x.shape[0])
train_x = train_x[inds]
train_y = train_y[inds]
x_labelled = []
y_labelled = []

for j in range(NUM_CLASSES):
    x_labelled.append(train_x[train_y == j][:int(NUM_LABELLED / NUM_CLASSES)])
    y_labelled.append(train_y[train_y == j][:int(NUM_LABELLED / NUM_CLASSES)])

x_labelled = np.concatenate(x_labelled, axis=0)
y_labelled = np.concatenate(y_labelled, axis=0)
del train_x

if True:
    print('Size of training data', x_labelled.shape[0], x_unlabelled.shape[0])
    # y_order = np.argsort(y_labelled)
    # _x_mean = x_labelled[y_order]
    # image = paramgraphics.mat_to_img(_x_mean.T, dim_input,
    #                                  tile_shape=(num_classes, num_labelled/num_classes),
    #                                  colorImg=colorImg, scale=generation_scale,
    #                                 save_path=os.path.join(outfolder, 'x_l_'+str(ssl_data_seed)+'_sgan.png'))

num_batches_l = x_labelled.shape[0] / BATCH_SIZE
num_batches_u = x_unlabelled.shape[0] / BATCH_SIZE
num_batches_e = eval_x.shape[0] / BATCH_SIZE_EVAL
rng = np.random.RandomState(NP_SEED)


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


# generator y2x: p_g(x, y) = p(y) p_g(x | y) where x = G(z, y), z follows p_g(z)
class Generator(nn.Module):
    def __init__(self, input_size=101, num_classes=10,
                 dense_neurons=8192,
                 num_output_features=3072):

        super(Generator, self).__init__()

        self.num_classes = num_classes

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

    def forward(self, z, y):
        x = mlp_concat(z, y, self.num_classes)

        x1 = self.Dense(x)
        x1 = self.Relu(x1)
        x1 = self.BatchNorm1D(x1)

        x2 = x1.resize_(-1, 512, 4, 4)
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

        x7 = wn(x6)

        return x7


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
        # self.ConvConcat = ConvConcatLayer()
        # self.MLPConcat = MLPConcatLayer()
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

        self.globalPool = nn.AdaptiveAvgPool2d(output_size=[16, 16])

        # MLPConcat

        self.lin = nn.Linear(in_features=256,
                             out_features=1)
        # smg

        if weight_init:
            # initialize weights for all conv and lin layers
            self.apply(init_weights(normal=[1.0, 0.05]))
            # log network structure
            self.logger.info(self)

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


# classifier module
class Classifier_Net(nn.Module):
    def __init__(self):
        super(Classifier_Net, self).__init__()

        self.conv1a = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3,
                                stride=1, padding=1)
        self.convWN = MeanOnlyBatchNorm(128)
        self.conv1b = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,
                                stride=1, padding=1)
        self.conv_relu = nn.LeakyReLU(negative_slope=0.1)
        self.conv1c = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,
                                stride=1, padding=1)
        self.conv_maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout2d(p=0.5)
        self.conv2a = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                                stride=1, padding=1)
        self.conv2b = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                                stride=1, padding=1)
        self.conv2c = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                                stride=1, padding=1)
        self.conv_maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout2d(p=0.5)
        self.conv3a = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,
                                stride=1, padding=0)
        self.conv3b = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1,
                                stride=1, padding=0)
        self.conv3c = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1,
                                stride=1, padding=0)
        self.conv_maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.dense = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = Gaussian_NoiseLayer(x.shape)
        x = self.convWN(self.conv_relu(self.conv1a(x)))
        x = self.convWN(self.conv_relu(self.conv1b(x)))
        x = self.convWN(self.conv_relu(self.conv1c(x)))
        x = self.conv_maxpool1(x)
        x = self.dropout1(x)
        x = self.convWN(self.conv_relu(self.conv2a(x)))
        x = self.convWN(self.conv_relu(self.conv2b(x)))
        x = self.convWN(self.conv_relu(self.conv2c(x)))
        x = self.conv_maxpool2(x)
        x = self.dropout2(x)
        x = self.convWN(self.conv_relu(self.conv3a(x)))
        x = self.convWN(self.conv_relu(self.conv3b(x)))
        x = self.convWN(self.conv_relu(self.conv3c(x)))
        x = self.conv_maxpool3(x)
        x = self.dense(x)
        return x


class Gaussian_NoiseLayer(nn.Module):
    def __init__(self, shape, std=0.15):
        super(Gaussian_NoiseLayer).__init__()
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


# inference module
class Inference_Net(nn.Module):
    def __init__(self):
        super(Inference_Net, self).__init__()
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
        self.inf4 = nn.Linear(in_features=512, out_features=n_z)

    def forward(self, x):
        x = F.leaky_relu(self.inf03(self.inf02(x)))
        x = F.leaky_relu(self.inf12(self.inf11(x)))
        x = F.leaky_relu(self.inf22(self.inf21(x)))
        x = F.leaky_relu(self.inf32(self.inf31(x)))
        x = self.inf4(x)

        return x


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
        # self.ConvConcat = ConvConcatLayer()
        # self.MLPConcat = MLPConcatLayer()
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
        self.lin_f0 = nn.Linear(in_features=512,
                                out_features=1024)
        # LReLU

        self.lin_f1 = nn.Linear(in_features=1024,
                                out_features=1)
        # smg

        if weight_init:
            # initialize weights for all conv and lin layers
            self.apply(init_weights(normal=[1.0, 0.05]))
            # log network structure
            self.logger.info(self)

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
        fusion = torch.cat(x_out.view(dims[0], dims[1], -1).squeeze(-1), z_out)

        f_out = self.LReLU(self.lin_f0(fusion))
        out = self.sgmd(self.lin_f1(f_out))

        return out


#########################################################################################################

### INIT ###

# GENRATOR

# INFERENCE

# CLASSIFIER

# DISCRIMINATOR

# LOSS FUNCTIONS
losses = {
    'bce': nn.BCELoss(),
    'mse': nn.MSELoss(),
    'ce': nn.CrossEntropyLoss()
}

# OPTIMIZERS
b1_c = rampdown_value * 0.9 + (1.0 - rampdown_value) * 0.5
optimizers = {
    'dis': optim.Adam(discriminator1.parameters() + discriminator2.parameters(), betas=(B1, 0.999), lr=LR),
    'gen': optim.Adam(generator.parameters(), betas=(B1, 0.999), lr=LR),
    'cla': optim.Adam(classifier.parameters(), betas=(b1_c, 0.999), lr=LR_CLA),  # robust adam??
    'cla_pre': optim.Adam(pre_classifier.parameters(), lr=LR_CLA, betas=(b1_c, 0.999)),
    'inf': optim.Adam(inference.parameters(), betas=(B1, 0.999), lr=LR)
}


#########################################################################################################

### PRETRAIN CLASSIFIER ###

print('Start pretraining')
for epoch in range(1, 1+NUM_EPOCHS_PRE):
    # randomly permute data and labels
    p_l = rng.permutation(x_labelled.shape[0])
    x_labelled = x_labelled[p_l]
    y_labelled = y_labelled[p_l]
    p_u = rng.permutation(x_unlabelled.shape[0]).astype('int32')

    x_labelled = Variable(x_labelled)
    y_labelled = Variable(y_labelled)

    eval_x = Variable(eval_x)
    eval_y = Variable(eval_y)

    whitener = zca(x = x_unlabelled)
    x_unlabelled = Variable(x_unlabelled)

    for i in range(num_batches_u):
        i_c = i % num_batches_l
        #bp = pretrain_batch_cla(x_labelled[i_c*batch_size:(i_c+1)*batch_size],
         #                       y_labelled[i_c*batch_size:(i_c+1)*batch_size],
          #                      p_u[i*batch_size:(i+1)*batch_size], 3e-3, 0.9, 100)

        #print(bp)
        x_l = x_labelled[i_c * batch_size:(i_c + 1) * batch_size]
        x_l_zca = whitener.apply(x_l)
        y = y_labelled[i_c*batch_size:(i_c+1)*batch_size]
        cla_out_y_l = classifier(x_l_zca)
        cla_cost_l = cross_entropy(cla_out_y_l, y)

        x_u_rep = x_unlabelled[p_u[i*batch_size:(i+1)*batch_size]] # sym_x_u_rep: shared_unlabel[slice_x_u_c]
        x_u_rep_zca = whitener.apply(x_u_rep)
        cla_out_y_rep = classifier(x_u_rep_zca)

        x_u = x_unlabelled[p_u[i*batch_size:(i+1)*batch_size]]
        x_u_zca = whitener.apply(x_u)
        cla_out_y = classifier(x_u_zca)
        cla_cost_u = 100 * mse(cla_out_y, cla_out_y_rep)

        pretrain_cost = cla_cost_l + cla_cost_u

        cla_optimizer = optim.Adam(classifier.parameters(), betas=(0.9, 0.999),
                                   lr= 3e-3)  # they implement robust adam

        pretrain_cost.backward()
        cla_optimizer.step()

    # evaluate = theano.function(inputs=[sym_x_eval, sym_y], outputs=[accurracy_eval], givens=cla_avg_givens)

    accurracy=[]
    for i in range(num_batches_e):
        x_eval = eval_x[i*batch_size_eval:(i+1)*batch_size_eval]
        y_eval = eval_y[i*batch_size_eval:(i+1)*batch_size_eval]
        x_eval_zca = whitener.apply(x_eval)

        cla_out_y_eval = classifier(x_eval_zca)
        accurracy_batch = accuracy_score(y_eval, cla_out_y_eval)


        #accurracy_batch = evaluate(eval_x[i*batch_size_eval:(i+1)*batch_size_eval], eval_y[i*batch_size_eval:(i+1)*batch_size_eval])
        accurracy.append(accurracy_batch)
    accurracy=np.mean(accurracy)
    print(str(epoch) + ':Pretrain error: ' + str(1- accurracy))



### GAN TRAINING ###

print("Start training")
for epoch in range(1, 1+NUM_EPOCHS):
    start = time.time()

    # randomly permute data and labels
    p_l = rng.permutation(x_labelled.shape[0])
    x_labelled = x_labelled[p_l]
    y_labelled = y_labelled[p_l]
    p_u = rng.permutation(x_unlabelled.shape[0]).astype('int32')
    p_u_d = rng.permutation(x_unlabelled.shape[0]).astype('int32')
    p_u_i = rng.permutation(x_unlabelled.shape[0]).astype('int32')

    x_labelled = Variable(x_labelled)
    y_labelled = Variable(y_labelled)
    eval_x = Variable(eval_x)
    eval_y = Variable(eval_y)

    whitener = zca(x = x_unlabelled)


    if epoch < 500:
        if epoch % 50 == 1: # 4, 8, 12, 16
            batch_l = 200 - (epoch // 50 + 1) * 16
            batch_c = (epoch // 50 + 1) * 16
            batch_g = 1#(epoch // 50 + 1) * 10
    elif epoch < 1000 and epoch % 100 == 0:
        batch_l = 50
        batch_c = 140 - 10 * (epoch-500)/100
        batch_g = 10 + 10 * (epoch-500)/100

    running_cla_cost = 0.0
    if (epoch % eval_epoch == 0):
        # te
        rampup_value = rampup(epoch-1)
        rampdown_value = rampdown(epoch-1)
        lr_c = cla_lr
        b1_c = rampdown_value * 0.9 + (1.0 - rampdown_value) * 0.5
        unsup_weight = rampup_value * scaled_unsup_weight_max if epoch > 1 else 0.
        w_g = np.float32(min(float(epoch) / 300.0, 1.0))

        size_l = 100
        size_g = 100
        size_u = 100

        for i in range(num_batches_u * eval_epoch):

            i_l = i % (x_labelled.shape[0] // size_l)
            i_u = i % (x_unlabelled.shape[0] // size_u)

            y_real = np.int32(np.random.randint(10, size=size_g))
            z_real = np.random.uniform(size=(size_g, n_z)).astype(np.float32)

            # zero the parameter gradients
            cla_optimizer.zero_grad()

            x_l = x_labelled[i_l*size_l:(i_l+1)*size_l]
            y = y_labelled[i_l*size_l:(i_l+1)*size_l]

            x_l_zca = whitener.apply(x_l)
            cla_out_y_l = classifier(x_l_zca)
            cla_cost_l = cross_entropy(cla_out_y_l, y) # size_average in pytorch is by default

            cla_out_y_rep = classifier(x_u_rep_zca)
            cla_out_y = classifier(x_u_zca)
            cla_cost_u = unsup_weight * mse(cla_out_y, cla_out_y_rep)

            y_m = y_real
            z_m = z_real
            gen_out_x_m = generator(z= z_m, y = y_m)
            gen_out_x_m_zca = whitener.apply(gen_out_x_m)
            cla_out_y_m = classifier(gen_out_x_m_zca)
            cla_cost_g = cross_entropy(cla_out_y_m, y_m) * w_g

            cla_cost = cla_cost_l + cla_cost_u + cla_cost_g
            #cla_cost_list = [cla_cost, cla_cost_l, cla_cost_u, cla_cost_g]

            # updates of C
            # b1_c = rampdown_value * 0.9 + (1.0 - rampdown_value) * 0.5
            cla_optimizer = optim.Adam(classifier.parameters(), betas=(b1_c, 0.999),
                                       lr=cla_lr)  # they implement robust adam

            cla_cost.backward()
            cla_optimizer.step()

            if i_l == ((x_labelled.shape[0] // size_l) - 1):
                p_l = rng.permutation(x_labelled.shape[0])
                x_labelled = x_labelled[p_l]
                y_labelled = y_labelled[p_l]
            if i_u == (num_batches_u - 1):
                p_u = rng.permutation(x_unlabelled.shape[0]).astype('int32')

                # print statistics
            running_cla_cost += cla_cost.data[0]

        accurracy = []
        for i in range(num_batches_e):
            x_eval = eval_x[i * batch_size_eval:(i + 1) * batch_size_eval]
            y_eval = eval_y[i * batch_size_eval:(i + 1) * batch_size_eval]
            x_eval_zca = whitener.apply(x_eval)
            cla_out_y_eval = classifier(x_eval_zca)
            accurracy_batch = accuracy_score(y_eval, cla_out_y_eval)
            #accurracy_batch = evaluate(eval_x[i * batch_size_eval:(i + 1) * batch_size_eval],
            #                           eval_y[i * batch_size_eval:(i + 1) * batch_size_eval])
            accurracy.append(accurracy_batch)
        accurracy = np.mean(accurracy)
        print('ErrorEval=%.5f\n' % (1 - accurracy,))
        with open(logfile, 'a') as f:
            f.write(('ErrorEval=%.5f\n\n' % (1 - accurracy,)))


    for i in range(num_batches_u):
        i_l = i % (x_labelled.shape[0] // batch_l)

        from_u_i = i*batch_size
        to_u_i = (i+1)*batch_size
        from_u_d = i*batch_c
        to_u_d = (i+1) * batch_c
        from_l = i_l*batch_l
        to_l = (i_l+1)*batch_l

        sample_y = np.int32(np.repeat(np.arange(num_classes), batch_size/num_classes))
        y_real = np.int32(np.random.randint(10, size=batch_g))
        z_real = np.random.uniform(size=(batch_g, n_z)).astype(np.float32)
        z_rand = np.random.uniform(size=(batch_size, n_z)).astype(np.float32)


        tmp = time.time()

        

        dl_b = train_batch_dis(x_labelled[from_l:to_l], y_labelled[from_l:to_l], p_u_d[from_u_d:to_u_d], y_real, z_real, p_u_i[from_u_i:to_u_i], sample_y, lr)
        for j in xrange(len(dl)):
            dl[j] += dl_b[j]

        
        #rain_batch_inf = theano.function(inputs=[slice_x_u_i, sym_y_g, sym_lr],
        #                                  outputs=inf_cost_list, updates=inf_updates,
        #                                  givens={sym_x_u_i: shared_unlabel[slice_x_u_i]})


        il_b = train_batch_inf(p_u_i[from_u_i:to_u_i], sample_y, lr)
        x_u_i = x_unlabelled[p_u_i[from_u_i:to_u_i]]

        y_g = sample_y
        gen_out_x = generator(z_rand, y_g)
        inf_z = inferentor(x_u_i)
        inf_z_g = inferentor(gen_out_x)
        rz = mse(gen_out_x, inf_z_g)
        inf_cost_p_i = bce(disxz_out_p, torch.zeros(disxz_out_p.shape))
        inf_cost = inf_cost_p_i + rz
        inf_optimizer = optim.Adam(inferentor.parameters(), betas=(b1, 0.999), lr=lr)
        inf_cost.backward()
        inf_optimizer.step()

        #for j in xrange(len(il)):
        #    il[j] += il_b[j]

        #gl_b = train_batch_gen(sample_y, lr)
        #for j in xrange(len(gl)):
        #    gl[j] += gl_b[j]

        if i_l == ((x_labelled.shape[0] // batch_l) - 1):
            p_l = rng.permutation(x_labelled.shape[0])
            x_labelled = x_labelled[p_l]
            y_labelled = y_labelled[p_l]

    #for i in xrange(len(dl)):
    #    dl[i] /= num_batches_u
    #for i in xrange(len(gl)):
    #    gl[i] /= num_batches_u
    #for i in xrange(len(cl)):
    #    cl[i] /= num_batches_u

    if (epoch >= anneal_lr_epoch) and (epoch % anneal_lr_every_epoch == 0):
        lr = lr * anneal_lr_factor
        cla_lr *= anneal_lr_factor_cla

    t = time.time() - start

    line = "*Epoch=%d Time=%.2f LR=%.5f\n" % (epoch, t, lr) + "DisLosses: " + str(dl) + "\nGenLosses: " + \
           str(gl) + "\nInfLosses: " + str(il) + "\nClaLosses: " + str(cl)
    print(line)
    with open(logfile, 'a') as f:
        f.write(line + "\n")
