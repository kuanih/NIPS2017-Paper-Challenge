'''
SGAN PyTroch implementation of << ... >>
'''

### IMPORTS ###

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os.path
import utils

# initialize logger
import logging.config
import yaml
with open('./log_config.yaml') as file:
    Dict = yaml.load(file)    # load config file
    logging.config.dictConfig(Dict)    # import config

logger = logging.getLogger(__name__)
logger.info('PyTorch version: ' + str(torch.__version__))

# loss logger
loss_logger = logging.getLogger('loss_logger')
loss_logger.setLevel(logging.INFO)
fh = logging.FileHandler('./losses.log')
fh.setLevel(logging.INFO)
loss_logger.addHandler(fh)

# import SGAN utils
from layers import rampup, rampdown
from zca import ZCA
from models import Generator, InferenceNet, ClassifierNet, DConvNet1, DConvNet2
from trainGAN import pretrain_classifier, train_classifier, train_gan, eval_classifier


### GLOBAL PARAMS ###
BATCH_SIZE = 200
BATCH_SIZE_EVAL = 200
NUM_CLASSES = 10
NUM_LABELLED = 4000
SSL_SEED = 1
NP_SEED = 1234
CUDA = torch.cuda.is_available()
logger.info('Cuda = ' + str(CUDA))

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

# results and checkpoints
path_out = "./results"
if not os.path.exists(path_out):
    os.makedirs(path_out)

checkpoint_directory = "./checkpoints/"
if not os.path.exists(checkpoint_directory):
    os.makedirs(checkpoint_directory)

classifier_result = checkpoint_directory + "classifier_pretrained.pth"
EPOCH_SAVE_CHECKPOINTS = 25

### DATA ###
logger.info('Loading data...')
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

num_batches_l = int(x_labelled.shape[0] // BATCH_SIZE)
num_batches_u = int(x_unlabelled.shape[0] // BATCH_SIZE)
num_batches_e = int(eval_x.shape[0] // BATCH_SIZE_EVAL)
rng = np.random.RandomState(NP_SEED)

#########################################################################################################


### INITS ###

# GENRATOR
generator = Generator(input_size=110, num_classes=NUM_CLASSES, dense_neurons=(4 * 4 * 512))

# INFERENCE
inference = InferenceNet(in_channels=IN_CHANNELS, n_z=N_Z)

# CLASSIFIER
classifier = ClassifierNet(in_channels=IN_CHANNELS)

# DISCRIMINATOR
discriminator1 = DConvNet1(channel_in=IN_CHANNELS, num_classes=NUM_CLASSES)
discriminator2 = DConvNet2(n_z=N_Z, channel_in=IN_CHANNELS, num_classes=NUM_CLASSES)


# put on GPU
if CUDA:
    generator.cuda()
    inference.cuda()
    classifier.cuda()
    discriminator1.cuda()
    discriminator2.cuda()

# ZCA
whitener = ZCA(x=x_unlabelled)

# LOSS FUNCTIONS
if CUDA:
    losses = {
        'bce': nn.BCELoss().cuda(),
        'mse': nn.MSELoss().cuda(),
        'ce': nn.CrossEntropyLoss().cuda()
    }
else:
    losses = {
        'bce': nn.BCELoss(),
        'mse': nn.MSELoss(),
        'ce': nn.CrossEntropyLoss()
    }


#########################################################################################################

### PRETRAIN CLASSIFIER ###
if os.path.isfile(classifier_result):
    logger.info('Load pretrained classifier from disk')
    classifier.load_state_dict(torch.load(classifier_result))
else:
    logger.info('Start pretraining...')
    for epoch in range(1, 1+NUM_EPOCHS_PRE):

        # pretrain classifier net
        classifier = pretrain_classifier(x_labelled, x_unlabelled, y_labelled, eval_x, eval_y, num_batches_l,
                                         BATCH_SIZE, num_batches_u, classifier, whitener, losses, rng, CUDA)

        # evaluate
        accurracy = eval_classifier(num_batches_e, eval_x, eval_y, BATCH_SIZE_EVAL, whitener, classifier, CUDA)

        logger.info(str(epoch) + ':Pretrain error_rate: ' + str(1 - accurracy))

        torch.save(classifier.state_dict(), classifier_result)


### GAN TRAINING ###

# assign start values
lr_cla = LR_CLA
lr = LR
start_full = time.time()

logger.info("Start GAN training...")
for epoch in range(1, 1+NUM_EPOCHS):

    # OPTIMIZERS
    optimizers = {
        'dis': optim.Adam(list(discriminator1.parameters()) + list(discriminator2.parameters()), betas=(B1, 0.999), lr=lr),
        'gen': optim.Adam(generator.parameters(), betas=(B1, 0.999), lr=lr),
        'inf': optim.Adam(inference.parameters(), betas=(B1, 0.999), lr=lr)
    }

    # randomly permute data and labels each epoch
    p_l = rng.permutation(x_labelled.shape[0])
    x_labelled = x_labelled[p_l]
    y_labelled = y_labelled[p_l]

    # permuted slicer objects
    p_u = rng.permutation(x_unlabelled.shape[0]).astype('int32')
    p_u_d = rng.permutation(x_unlabelled.shape[0]).astype('int32')
    p_u_i = rng.permutation(x_unlabelled.shape[0]).astype('int32')

    # set epoch dependent values
    if epoch < (NUM_EPOCHS/2):
        if epoch % 50 == 1:
            batch_l = 200 - (epoch // 50 + 1) * 16
            batch_c = (epoch // 50 + 1) * 16
            batch_g = 1
    elif epoch < NUM_EPOCHS and epoch % 100 == 0:
        batch_l = 50
        batch_c = 140 - 10 * (epoch-500)/100
        batch_g = 10 + 10 * (epoch-500)/100

    # if current epoch is an evaluation epoch, train classifier and report results
    if epoch % EVAL_EPOCH == 0:

        logger.info('Train classifier...')

        rampup_value = rampup(epoch-1)
        rampdown_value = rampdown(epoch-1)
        b1_c = rampdown_value * 0.9 + (1.0 - rampdown_value) * 0.5
        unsup_weight = rampup_value * SCALED_UNSUP_WEIGHT_MAX if epoch > 1 else 0.0
        w_g = np.float32(min(float(epoch) / 300.0, 1.0))

        size_l = 100
        size_g = 100
        size_u = 100

        cla_losses = train_classifier(x_labelled=x_labelled,
                                      y_labelled=y_labelled,
                                      x_unlabelled=x_unlabelled,
                                      num_batches_u=num_batches_u,
                                      eval_epoch=EVAL_EPOCH,
                                      size_l=size_l,
                                      size_u=size_u,
                                      size_g=size_g,
                                      n_z=N_Z,
                                      whitener=whitener,
                                      classifier=classifier,
                                      p_u=p_u,
                                      unsup_weight=unsup_weight,
                                      losses=losses,
                                      generator=generator,
                                      w_g=w_g,
                                      cla_lr=lr_cla,
                                      rng=rng,
                                      b1_c=b1_c,
                                      cuda=CUDA)

        # evaluate & report
        accurracy = eval_classifier(num_batches_e, eval_x, eval_y, BATCH_SIZE_EVAL, whitener, classifier, CUDA)

        logger.info('Evaluation error_rate: %.5f\n' % (1 - accurracy))

    logger.info('Train generator, inference and discriminator model...')
    # train GAN model
    for i in range(num_batches_u):
        gan_losses = train_gan(discriminator1=discriminator1,
                               discriminator2=discriminator2,
                               generator=generator,
                               inferentor=inference,
                               classifier=classifier,
                               whitener=whitener,
                               x_labelled=x_labelled,
                               x_unlabelled=x_unlabelled,
                               y_labelled=y_labelled,
                               p_u_d=p_u_d,
                               p_u_i=p_u_i,
                               num_classes=NUM_CLASSES,
                               batch_size=BATCH_SIZE,
                               num_batches_u=num_batches_u,
                               batch_c=batch_c,
                               batch_l=batch_l,
                               batch_g=batch_g,
                               n_z=N_Z,
                               optimizers=optimizers,
                               losses=losses,
                               rng=rng,
                               cuda=CUDA)

    # anneal the learning rates
    if (epoch >= ANNEAL_EPOCH) and (epoch % ANNEAL_EVERY_EPOCH == 0):
        lr = lr * ANNEAL_FACTOR
        lr_cla *= ANNEAL_FACTOR_CLA

    # report and log training info
    t = time.time() - start_full
    line = "*Epoch=%d Time=%.2f LR=%.5f\n" % (epoch, t, lr) + "DisLosses: " + str(gan_losses['dis']) + "\nGenLosses: " + \
           str(gan_losses['gen']) + "\nInfLosses: " + str(gan_losses['inf']) + "\nClaLosses: " + str(cla_losses)
    logger.info(line)

    loss_logger.info(line)

    # save checkpoints
    if epoch % EPOCH_SAVE_CHECKPOINTS == 0 or epoch == NUM_EPOCHS:
        utils.save_checkpoint(generator, "generator", checkpoint_directory, epoch)
        utils.save_checkpoint(discriminator1, "discriminator1", checkpoint_directory, epoch)
        utils.save_checkpoint(discriminator2, "discriminator2", checkpoint_directory, epoch)
        utils.save_checkpoint(inference, "inferencer", checkpoint_directory, epoch)
