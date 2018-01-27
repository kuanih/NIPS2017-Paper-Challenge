'''
SGAN PyTroch implementation of << ... >>
'''


### IMPORTS ###

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import time
import utils
from sklearn.metrics import accuracy_score

# initialize logger
import logging.config
import yaml
with open('./log_config.yaml') as file:
    Dict = yaml.load(file)    # load config file
    logging.config.dictConfig(Dict)    # import config

logger = logging.getLogger(__name__)
logger.info('PyTorch version: ' + str(torch.__version__))

# import SGAN utils
from layers import rampup, rampdown
from zca import ZCA
from models import Generator, InferenceNet, ClassifierNet, DConvNet1, DConvNet2



### GLOBAL PARAMS ###
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


### DATA ###
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
    logger.info('Size of training data', x_labelled.shape[0], x_unlabelled.shape[0])
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

#########################################################################################################


### INITS ###

# GENRATOR
generator = Generator(input_size=101, num_classes=NUM_CLASSES, dense_neurons=8192, num_output_features=3072)

# INFERENCE
inference = InferenceNet(in_channels=IN_CHANNELS, n_z=N_Z)

# CLASSIFIER
classifier = ClassifierNet(in_channels=IN_CHANNELS)

# DISCRIMINATOR
discriminator1 = DConvNet1(channel_in=IN_CHANNELS, num_classes=NUM_CLASSES)
discriminator2 = DConvNet2(n_z=N_Z, channel_in=IN_CHANNELS, num_classes=NUM_CLASSES)

# ZCA
whitener = ZCA(x=x_unlabelled)

# LOSS FUNCTIONS
losses = {
    'bce': nn.BCELoss(),
    'mse': nn.MSELoss(),
    'ce': nn.CrossEntropyLoss()
}


#########################################################################################################

### PRETRAIN CLASSIFIER ###

logger.info('Start pretraining')
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

        #logger.info(bp)
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
    logger.info(str(epoch) + ':Pretrain error: ' + str(1- accurracy))



### GAN TRAINING ###
lr_cla = LR_CLA
lr = LR

logger.info("Start GAN training")
for epoch in range(1, 1+NUM_EPOCHS):

    start_full = time.time()

    # OPTIMIZERS
    optimizers = {
        'dis': optim.Adam(discriminator1.parameters() + discriminator2.parameters(), betas=(B1, 0.999), lr=lr),
        'gen': optim.Adam(generator.parameters(), betas=(B1, 0.999), lr=lr),
        'inf': optim.Adam(inference.parameters(), betas=(B1, 0.999), lr=lr)
    }

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

    if epoch < 500:
        if epoch % 50 == 1:
            batch_l = 200 - (epoch // 50 + 1) * 16
            batch_c = (epoch // 50 + 1) * 16
            batch_g = 1
    elif epoch < 1000 and epoch % 100 == 0:
        batch_l = 50
        batch_c = 140 - 10 * (epoch-500)/100
        batch_g = 10 + 10 * (epoch-500)/100

    total_cla_cost = 0.0

    if epoch % EVAL_EPOCH == 0:

        rampup_value = rampup(epoch-1)
        rampdown_value = rampdown(epoch-1)
        lr_c = lr_cla
        b1_c = rampdown_value * 0.9 + (1.0 - rampdown_value) * 0.5
        unsup_weight = rampup_value * SCALED_UNSUP_WEIGHT_MAX if epoch > 1 else 0.0
        w_g = np.float32(min(float(epoch) / 300.0, 1.0))

        size_l = 100
        size_g = 100
        size_u = 100

        for i in range(num_batches_u * EVAL_EPOCH):

            cla_losses = train_classifier(...)
            total_cla_cost += cla_losses


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
        logger.info('ErrorEval=%.5f\n' % (1 - accurracy,))


    start_gan = time.time()

    for i in range(num_batches_u):

        gan_losses = train_gan(...)

        # ??
        if i_l == ((x_labelled.shape[0] // batch_l) - 1):
            p_l = rng.permutation(x_labelled.shape[0])
            x_labelled = x_labelled[p_l]
            y_labelled = y_labelled[p_l]

    # annealing the learning rates
    if (epoch >= ANNEAL_EPOCH) and (epoch % ANNEAL_EVERY_EPOCH == 0):
        lr = lr * ANNEAL_FACTOR
        lr_cla *= ANNEAL_FACTOR_CLA



    t = time.time() - start

    line = "*Epoch=%d Time=%.2f LR=%.5f\n" % (epoch, t, lr) + "DisLosses: " + str(dl) + "\nGenLosses: " + \
           str(gl) + "\nInfLosses: " + str(il) + "\nClaLosses: " + str(cl)
    logger.info(line)





