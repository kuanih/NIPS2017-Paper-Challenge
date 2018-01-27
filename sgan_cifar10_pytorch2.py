import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
import numpy as np
import time
import math
import utils
from layers import conv_concat, mlp_concat
from sklearn.metrics import accuracy_score
import os
import zca


outfolder="./cifar10_results"
logfile=os.path.join(outfolder, 'logfile.log')
'''
data
'''



batch_size = 200
batch_size_eval = 200
num_classes = 10
num_labelled = 4000
ssl_data_seed = 1

train_x, train_y = utils.load('./cifar10/', 'train')
eval_x, eval_y = utils.load('./cifar10/', 'test')

train_y = np.int32(train_y)
eval_y = np.int32(eval_y)
x_unlabelled = train_x.copy()

rng_data = np.random.RandomState(ssl_data_seed)
inds = rng_data.permutation(train_x.shape[0])
train_x = train_x[inds]
train_y = train_y[inds]
x_labelled = []
y_labelled = []

for j in range(num_classes):
    x_labelled.append(train_x[train_y == j][:int(num_labelled / num_classes)])
    y_labelled.append(train_y[train_y == j][:int(num_labelled / num_classes)])

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

num_batches_l = x_labelled.shape[0] / batch_size
num_batches_u = x_unlabelled.shape[0] / batch_size
num_batches_e = eval_x.shape[0] / batch_size_eval

'''
parameters
'''
seed=1234
rng=np.random.RandomState(seed)

# data dependent
in_channels = 3

# evaluation
vis_epoch=1
eval_epoch=1

# C
scaled_unsup_weight_max = 100.0
# G
n_z=100
# optimization
b1=.5 # mom1 in Adam
batch_size=200
batch_size_eval=200
lr=3e-4
cla_lr=3e-3
num_epochs=1000
pre_num_epoch=20
anneal_lr_epoch=200
anneal_lr_every_epoch=1
anneal_lr_factor_cla=.99
anneal_lr_factor=.995

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

        x7 = nn.utils.weight_norm(x6)

        return x7
generator = Generator()

# discriminator xy2p: test a pair of input comes from p(x, y) instead of p_c or p_g
class Discriminator(nn.Module):
    def _init__(self):
        super(Discriminator, self).__init__()

    def forward(self, x):
        return x

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

classifier = Classifier_Net()

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

# discriminator xz
class Discriminator_xz(nn.Module):
    def __init__(self):
        super(Discriminator_xz, self).__init__()

    def forward(self, x):
        return x

'''
objectives
'''
# zca

# init

# outputs

# costs
cross_entropy = nn.CrossEntropyLoss()
mse = nn.MSELoss()
bce = nn.BCELoss()

dis_cost_p = bce(dis_out_p, torch.ones(dis_out_p.shape)) # D distincts p
dis_cost_p_g = bce(dis_out_p_g, torch.zeros(dis_out_p_g.shape)) # D distincts p_g
gen_cost_p_g_1 = bce(dis_out_p_g, torch.ones(dis_out_p_g.shape)) # G fools D

disxz_cost_p = bce(disxz_out_p, torch.ones(disxz_out_p.shape))
disxz_cost_p_g = bce(disxz_out_p_g, torch.zeros(disxz_out_p_g.shape))
inf_cost_p_i = bce(disxz_out_p, torch.zeros(disxz_out_p.shape))
gen_cost_p_g_2 = bce(disxz_out_p_g, torch.ones(disxz_out_p_g.shape))


# cost of labeled data is the difference between the classifed outputs and the targets
#cla_cost_l = cross_entropy(cla_out_y_l, y) # size_average in pytorch is by default
# cost of unlabeled data
#cla_cost_u = unsup_weight * mse(cla_out_y, cla_out_y_rep)
#cla_cost_g = cross_entropy(cla_out_y_m, y_m) * w_g

rz = mse(inf_z_g, z_rand)
ry = cross_entropy(cla_out_y_g, y_g)

pretrain_cost = cla_cost_l + cla_cost_u

#la_cost = cla_cost_l + cla_cost_u + cla_cost_g

dis_cost = dis_cost_p + dis_cost_p_g
disxz_cost = disxz_cost_p + disxz_cost_p_g
inf_cost = inf_cost_p_i + rz
gen_cost = gen_cost_p_g_1 + gen_cost_p_g_2 + rz + ry

dis_cost_list=[dis_cost + disxz_cost, dis_cost, dis_cost_p, dis_cost_p_g, disxz_cost, disxz_cost_p, disxz_cost_p_g]
gen_cost_list=[gen_cost, gen_cost_p_g_1, gen_cost_p_g_2, rz, ry]
inf_cost_list=[inf_cost, inf_cost_p_i, rz]

#cla_cost_list=[cla_cost, cla_cost_l, cla_cost_u, cla_cost_g]

inf_cost_list=[inf_cost, inf_cost_p_i, rz]

# updates of D
dis_optimizer = optim.Adam(discriminator.parameters()+ discriminator_xz.parameters(), betas= (b1, 0.999), lr = lr) # just adding parameters togehter?
# updates of G
gen_optimizer = optim.Adam(generator.parameters(), betas= (b1, 0.999), lr = lr)
# updates of C
# b1_c = rampdown_value * 0.9 + (1.0 - rampdown_value) * 0.5
cla_optimizer = optim.Adam(classifier.parameters(), betas=(b1_c, 0.999),lr = cla_lr) # they implement robust adam
pretrain_cla_optimizer = optim.Adam(pre_classifier.parameters(), lr = cla_lr, betas=(b1_c, 0.999))
# updates of I
inf_optimizer = optim.Adam(inference.parameters(), betas= (b1, 0.999), lr = lr)

'''
Pretrain C
'''
print('Start pretraining')
for epoch in range(1, 1+pre_num_epoch):
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
    x_unlalbelled = Variable(x_unlabelled)

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


'''
Training
'''
print("Start training")
for epoch in range(1, 1+num_epochs):
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
            y_predicted = classifier(x_eval)

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

        tmp = time.time()

        

        dl_b = train_batch_dis(x_labelled[from_l:to_l], y_labelled[from_l:to_l], p_u_d[from_u_d:to_u_d], y_real, z_real, p_u_i[from_u_i:to_u_i], sample_y, lr)
        for j in xrange(len(dl)):
            dl[j] += dl_b[j]

        il_b = train_batch_inf(p_u_i[from_u_i:to_u_i], sample_y, lr)
        for j in xrange(len(il)):
            il[j] += il_b[j]

        gl_b = train_batch_gen(sample_y, lr)
        for j in xrange(len(gl)):
            gl[j] += gl_b[j]

        if i_l == ((x_labelled.shape[0] // batch_l) - 1):
            p_l = rng.permutation(x_labelled.shape[0])
            x_labelled = x_labelled[p_l]
            y_labelled = y_labelled[p_l]



