'''
Discriminators file for Structured GAN
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import weight_norm as wn    # import weight_norm

from layers import mlp_concat, conv_concat, init_weights

import logging


### discriminators ###

# D1
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
        #self.ConvConcat = ConvConcatLayer()
        #self.MLPConcat = MLPConcatLayer()
        self.LReLU = nn.LeakyReLU(negative_slope=0.2)   # leaky ReLU activation function
        self.sgmd = nn.Sigmoid()     # sigmoid activation function
        self.drop = nn.Dropout(p=p_dropout)   # dropout layer

        # input -->
            # drop
            # ConvConcat

        self.conv1 = wn(nn.Conv2d(in_channels=channel_in, out_channels=32,
                                  kernel_size=(3, 3), stride=(1,1), padding=1, bias=False))
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
        x0 = conv_concat(x0, y, self.num_classes)
        
        x1 = self.LReLU(self.conv1(x0))
        x1 = conv_concat(x1, y, self.num_classes)
        
        x2 = self.LReLU(self.conv2(x1))
        x2 = self.drop(x2)
        x2 = conv_concat(x2, y, self.num_classes)
        
        x3 = self.LReLU(self.conv3(x2))
        x3 = conv_concat(x3, y, self.num_classes)

        x4 = self.LReLU(self.conv4(x3))
        x4 = self.drop(x4)
        x4 = conv_concat(x4, y, self.num_classes)

        x5 = self.LReLU(self.conv5(x4))
        x5 = conv_concat(x5, y, self.num_classes)

        x6 = self.LReLU(self.conv6(x5))

        x_pool = self.globalPool(x6)
        x_out = self.mlp_concat(x_pool, y)

        out = self.sgmd(self.lin(x_out))

        return out


# D2
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


dis0 = DConvNet1(3, 10, weight_init=False)
a = Variable(torch.randn(2, 3, 32, 32))
b = Variable(torch.randn(2, 1))

dis0(a, b)


### objectives ###
'''
# zca
whitener = ZCA(x=x_unlabelled)
sym_x_l_zca = whitener.apply(sym_x_l)
sym_x_eval_zca = whitener.apply(sym_x_eval)
sym_x_u_zca = whitener.apply(sym_x_u)
sym_x_u_rep_zca = whitener.apply(sym_x_u_rep)
sym_x_u_d_zca = whitener.apply(sym_x_u_d)

# init
init_fn = theano.function([sym_x_u], [], updates=init_updates)

# outputs
dis_out_p = ll.get_output(dis_layers[-1],
                          {dis_in_x:T.concatenate([sym_x_l, sym_x_u_d, gen_out_x_m], axis=0)[:batch_size],
                           dis_in_y:T.concatenate([sym_y, cla_out_y_d_hard, sym_y_m], axis=0)[:batch_size]},
                          deterministic=False)
dis_out_p_g = ll.get_output(dis_layers[-1], {dis_in_x:gen_out_x,dis_in_y:sym_y_g}, deterministic=False)

disxz_out_p = ll.get_output(disxz_layers[-1], {disxz_in_x:sym_x_u_i, disxz_in_z: inf_z}, deterministic=False)
disxz_out_p_g = ll.get_output(disxz_layers[-1], {disxz_in_x:gen_out_x, disxz_in_z: sym_z_rand}, deterministic=False)


# costs
bce = lasagne.objectives.binary_crossentropy

dis_cost_p = bce(dis_out_p, T.ones(dis_out_p.shape)).mean() # D distincts p
dis_cost_p_g = bce(dis_out_p_g, T.zeros(dis_out_p_g.shape)).mean() # D distincts p_g

disxz_cost_p = bce(disxz_out_p, T.ones(disxz_out_p.shape)).mean()
disxz_cost_p_g = bce(disxz_out_p_g, T.zeros(disxz_out_p_g.shape)).mean()


rz = mean_squared_error(inf_z_g, sym_z_rand, n_z)
ry = categorical_crossentropy(cla_out_y_g, sym_y_g)

dis_cost = dis_cost_p + dis_cost_p_g
disxz_cost = disxz_cost_p + disxz_cost_p_g

dis_cost_list=[dis_cost + disxz_cost, dis_cost, dis_cost_p, dis_cost_p_g, disxz_cost, disxz_cost_p, disxz_cost_p_g]

# updates of D
dis_params = ll.get_all_params(dis_layers, trainable=True) + ll.get_all_params(disxz_layers, trainable=True)
dis_grads = T.grad(dis_cost+disxz_cost, dis_params)
dis_updates = lasagne.updates.adam(dis_grads, dis_params, beta1=b1, learning_rate=sym_lr)


# function compile
train_batch_dis = theano.function(inputs=[sym_x_l, sym_y, slice_x_u_d, sym_y_m, sym_z_m, slice_x_u_i, sym_y_g, sym_lr],
                                  outputs=dis_cost_list, updates=dis_updates,
                                  givens={sym_x_u_d: shared_unlabel[slice_x_u_d], sym_x_u_i: shared_unlabel[slice_x_u_i]})
'''



### train and evaluate ###
'''
init_fn(x_unlabelled[:batch_size])



for epoch in range(1, 1+num_epochs):
    start = time.time()

    # randomly permute data and labels
    p_l = rng.permutation(x_labelled.shape[0])
    x_labelled = x_labelled[p_l]
    y_labelled = y_labelled[p_l]
    p_u = rng.permutation(x_unlabelled.shape[0]).astype('int32')
    p_u_d = rng.permutation(x_unlabelled.shape[0]).astype('int32')
    p_u_i = rng.permutation(x_unlabelled.shape[0]).astype('int32')

    if epoch < 500:
        if epoch % 50 == 1: # 4, 8, 12, 16
            batch_l = 200 - (epoch // 50 + 1) * 16
            batch_c = (epoch // 50 + 1) * 16
            batch_g = 1#(epoch // 50 + 1) * 10
    elif epoch < 1000 and epoch % 100 == 0:
        batch_l = 50
        batch_c = 140 - 10 * (epoch-500)/100
        batch_g = 10 + 10 * (epoch-500)/100

    dl = [0.] * len(dis_cost_list)
    gl = [0.] * len(gen_cost_list)
    cl = [0.] * len(cla_cost_list)
    il = [0.] * len(inf_cost_list)



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

    for i in xrange(len(dl)):
        dl[i] /= num_batches_u
    for i in xrange(len(gl)):
        gl[i] /= num_batches_u
    for i in xrange(len(cl)):
        cl[i] /= num_batches_u
'''




