'''
Training routines for SGAN
'''

import torch.optim as optim
from sklearn.metrics import accuracy_score
import torch
from torch.autograd import Variable
import numpy as np


def pretrain_classifier(x_labelled, x_unlabelled, y_labelled, eval_x, eval_y, num_batches_l,
                        batch_size, num_batches_u, classifier, whitener, losses, rng, cuda):

    # randomly permute data and labels
    permutation_labelled = rng.permutation(x_labelled.shape[0])
    x_labelled = x_labelled[permutation_labelled]
    y_labelled = y_labelled[permutation_labelled]
    permutation_unlabelled = rng.permutation(x_unlabelled.shape[0]).astype('int32')

    x_labelled = Variable(torch.from_numpy(x_labelled))
    y_labelled = Variable(torch.from_numpy(y_labelled))

    eval_x = Variable(torch.from_numpy(eval_x))
    eval_y = Variable(torch.from_numpy(eval_y))
    x_unlabelled = Variable(torch.from_numpy(x_unlabelled))

    if cuda:
        x_labelled, y_labelled, eval_x, eval_y, x_unlabelled = \
            x_labelled.cuda(), y_labelled.cuda(), eval_x.cuda(), eval_y.cuda(), x_unlabelled.cuda()

    for i in range(num_batches_u):
        i_c = i % num_batches_l
        x_l = x_labelled[i_c * batch_size:(i_c + 1) * batch_size]
        x_l_zca = whitener.apply(x_l)
        y = y_labelled[i_c * batch_size:(i_c + 1) * batch_size]
        y = y.type(torch.LongTensor)
        if cuda:
            y = y.cuda()

        # classify input
        cla_out_y_l = classifier(x_l_zca, cuda=cuda)
        # calculate loss
        cla_cost_l = losses['ce'](cla_out_y_l, y)

        batch_slicer = torch.from_numpy(permutation_unlabelled[i * batch_size:(i + 1) * batch_size]).type(torch.LongTensor)
        if cuda:
            batch_slicer = batch_slicer.cuda()
        x_u_rep = x_unlabelled[batch_slicer]
        x_u_rep_zca = whitener.apply(x_u_rep)
        # classify input
        cla_out_y_rep = classifier(x_u_rep_zca, cuda=cuda)
        target = cla_out_y_rep.detach()
        del cla_out_y_rep

        x_u = x_unlabelled[batch_slicer]
        x_u_zca = whitener.apply(x_u)
        # classify input
        cla_out_y = classifier(x_u_zca, cuda=cuda)

        # calculate loss
        cla_cost_u = 100 * losses['mse'](cla_out_y, target)

        # sum losses
        pretrain_cost = cla_cost_l + cla_cost_u

        # run optimization and update weights
        cla_optimizer = optim.Adam(classifier.parameters(), betas=(0.9, 0.999), lr=3e-3)  # they implement robust adam
        pretrain_cost.backward()
        cla_optimizer.step()

    return classifier


def train_discriminator(discriminator1, discriminator2, generator, inferentor, classificator, whitener,
                        x_labelled, x_unlabelled, y_labelled,
                        slice_x_dis, y_real, z_real, slice_x_inf, sample_y, z_rand,
                        batch_size, optimizer, loss, cuda):
    '''

    Args:
        discriminator1(DConvNet1): Discriminator instance xy
        discriminator2(DConvNet2): Discriminator instance xz
        generator(Generator): Generator instance
        inferentor(InferenceNet): Inference Net instance
        classificator(ClassifierNet): Classifier Net instance
        whitener(ZCA): ZCA instance
        x_labelled: batch of labelled input data
        x_unlabelled: batch of unlabelled input data
        y_labelled: batch of corresponding labels
        slice_x_dis: indexes to select unlabelled data for discriminator
        y_real: class labels
        z_real: generator_x_m noise input
        slice_x_inf: indexes to select unlabelled data for inference net
        sample_y: sampled labels
        z_rand: generator_x noise input
        batch_size(int): size of mini-batch
        d1_optimizer(torch.optim): optimizer instance for discriminator1
        d2_optimizer(torch.optim): optimizer instance for discriminator2
        loss(torch.nn.Loss): loss instance for discriminators (BCE)
        cuda(bool): cuda flag (GPU)

    Returns: list(discriminator1 loss, discriminator2 loss)

    '''

    '''
    Parameter Translation: Theano original --> PyTorch
    input:
        x_labelled[from_l:to_l],  # sym_x_l
        y_labelled[from_l:to_l],  # sym_y
        p_u_d[from_u_d:to_u_d] --> slice_x_dis,  # slice_x_u_d
        y_real,  # sym_y_m
        z_real,  # sym_z_m
        p_u_i[from_u_i:to_u_i] --> slice_x_inf,  # slice_x_u_i
        sample_y,  # sym_y_g
    '''


    # get respective data slices for batch
    unlabel_dis = x_unlabelled[slice_x_dis]  # original: sym_x_u_d
    unlabel_dis_zca = whitener.apply(unlabel_dis)  # original: sym_x_u_d_zca
    unlabel_inf = x_unlabelled[slice_x_inf]  # original: sym_x_u_i

    # convert data ndarrays to pytorch tensor variables
    x_labelled = Variable(torch.from_numpy(x_labelled))
    unlabel_dis = Variable(torch.from_numpy(unlabel_dis))
    unlabel_dis_zca = Variable(torch.from_numpy(unlabel_dis_zca))
    unlabel_inf = Variable(torch.from_numpy(unlabel_inf))

    if cuda:
        x_labelled, unlabel_dis, \
        unlabel_dis_zca, unlabel_inf = x_labelled.cuda(), unlabel_dis.cuda(), \
                                       unlabel_dis_zca.cuda(), unlabel_inf.cuda()

    # generate samples
    gen_out_x = generator(sample_y, z_rand)
    gen_out_x_m = generator(y_real, z_real)

    # compute inference
    inf_z = inferentor(unlabel_inf)

    # classify
    cla_out = classificator(unlabel_dis_zca)
    cla_out_val, cla_out_idx = cla_out.max(dim=1)

    # concatenate inputs
    x_in = torch.cat([x_labelled, unlabel_dis, gen_out_x_m], dim=0)[:batch_size]
    y_in = torch.cat([y_labelled, cla_out_idx, y_real], dim=0)[:batch_size]

    # calculate probabilities by discriminators
    dis1_out_p = discriminator1(x_in, y_in)
    dis1_out_pg = discriminator1(gen_out_x, sample_y)

    dis2_out_p = discriminator2(unlabel_inf, inf_z)
    dis2_out_pg = discriminator2(z=z_rand, x=gen_out_x)

    # create discriminator labels
    p_label_d1 = Variable(torch.ones(dis1_out_p.size()))
    pg_label_d1 = Variable(torch.zeros(dis1_out_pg.size()))
    p_label_d2 = Variable(torch.ones(dis2_out_p.size()))
    pg_label_d2 = Variable(torch.zeros(dis2_out_pg.size()))

    if cuda:
        p_label_d1, pg_label_d1, \
        p_label_d2, pg_label_d2 = p_label_d1.cuda(), pg_label_d1.cuda(), \
                                  p_label_d2.cuda(), pg_label_d2.cuda()

    # compute loss
    dis1_cost_p = loss(dis1_out_p, p_label_d1)
    dis1_cost_pg = loss(dis1_out_pg, pg_label_d1)
    dis2_cost_p = loss(dis2_out_p, p_label_d2)
    dis2_cost_pg = loss(dis2_out_pg, pg_label_d2)

    # sum individual losses
    dis1_cost = dis1_cost_p + dis1_cost_pg  # for report
    dis2_cost = dis2_cost_p + dis2_cost_pg  # for report
    total_cost = dis1_cost + dis2_cost

    # optimization routines and weight updates
    optimizer.zero_grad()
    total_cost.backward()
    optimizer.step()

    return total_cost.cpu().numpy().mean()


def train_inferentor(x_unlabelled, sample_y, generator, z_rand, discriminator2, inferentor,
                     mse, bce, slice_x_u_i, optimizer, cuda):

    x_u_i = x_unlabelled[slice_x_u_i]
    x_u_i = Variable(torch.from_numpy(x_u_i))

    if cuda:
        x_u_i = x_u_i.cuda()

    y_g = sample_y
    gen_out_x = generator(z_rand, y_g)
    inf_z = inferentor(x_u_i)
    inf_z_g = inferentor(gen_out_x)
    disxz_out_p = discriminator2(z=inf_z, x=x_u_i)
    rz = mse(gen_out_x, inf_z_g)

    inf_cost_p_i = bce(disxz_out_p, torch.zeros(disxz_out_p.shape))
    inf_cost = inf_cost_p_i + rz

    optimizer.zero_grad()
    inf_cost.backward()
    optimizer.step()
    return inf_cost.cpu().numpy().mean()


def train_generator(whitener, optimizer, BCE_loss, MSE_loss, cross_entropy_loss,
                    discriminator1, discriminator2, inferentor, generator, classifier, sample_y, z_rand):
    '''
    Args:
        whitener(ZCA):      ZCA instance
        optimizer:          optimizer  for generator
        BCE_loss:           binary cross entropy loss
        MSE_loss:           mean squared error loss
        cross_entropy_loss: cross entropy loss
        discriminator1(DConvNet1): Discriminator instance xy
        discriminator2(DConvNet2): Discriminator instance xz
        inferentor:          Inference net
        generator:          Generator net
        classifier:         Classificaiton net
        sample_y:           sampled labels
        z_rand:             random z sample


    Returns:

    '''
    # compute loss
    gen_out_x = generator(z_rand, sample_y)
    inf_z_g = inferentor(gen_out_x)
    gen_out_x_zca = whitener.apply(gen_out_x)
    cla_out_y_g = classifier(gen_out_x_zca)
    rz = MSE_loss(inf_z_g, z_rand)
    ry = cross_entropy_loss(cla_out_y_g, sample_y)
    dis_out_p_g = discriminator1(x=gen_out_x, y=sample_y)
    disxz_out_p_g = discriminator2(z=z_rand, x=gen_out_x)

    gen_cost_p_g_1 = BCE_loss(dis_out_p_g, torch.ones(dis_out_p_g.shape))
    gen_cost_p_g_2 = BCE_loss(disxz_out_p_g, torch.ones(disxz_out_p_g.shape))

    generator_cost = gen_cost_p_g_1 + gen_cost_p_g_2 + rz + ry

    # optimization routines and weight updates
    optimizer.zero_grad()
    generator_cost.backward()
    optimizer.step()

    return generator_cost.cpu().numpy().mean()


def train_classifier(x_labelled, y_labelled, x_unlabelled, num_batches_u, eval_epoch,
                     size_l, size_u, size_g, n_z, whitener, classifier, p_u,
                     unsup_weight, losses, generator, w_g, cla_lr, rng, b1_c, cuda):
    '''

    Args:
        x_labelled: batch of labelled input data
        y_labelled: batch of labels
        x_unlabelled: unlabelled data
        num_batches_u:
        eval_epoch:
        size_l:
        size_u:
        size_g:
        n_z:
        whitener:
        classifier: Classifier Net instance
        p_u:
        unsup_weight:
        losses(dict): dictionary containing respective loss instances (BCE, MSE, CE)
        generator:
        w_g:
        cla_lr:
        rng:
        b1_c:

    Returns:

    '''

    running_cla_cost = 0.0
    epochs = num_batches_u * eval_epoch

    for i in range(epochs):

        i_l = i % (x_labelled.shape[0] // size_l)
        i_u = i % (x_unlabelled.shape[0] // size_u)

        y_real = np.int32(np.random.randint(10, size=size_g))
        z_real = np.random.uniform(size=(size_g, n_z)).astype(np.float32)

        x_l = x_labelled[i_l * size_l:(i_l + 1) * size_l]
        y = y_labelled[i_l * size_l:(i_l + 1) * size_l]
        x_l_zca = whitener.apply(x_l)

        slice_x_u_c = p_u[i_u*size_u:(i_u+1)*size_u]
        x_u_rep = x_unlabelled[slice_x_u_c]  # copy x_u_zca? double assigned variable??
        x_u = x_unlabelled[slice_x_u_c]
        x_u_rep_zca = whitener.apply(x_u_rep)
        x_u_zca = whitener.apply(x_u)

        # convert to torch tensor variable
        y_real = Variable(torch.from_numpy(y_real))
        z_real = Variable(torch.from_numpy(z_real))
        y = Variable(torch.from_numpy(y).type(torch.LongTensor))
        x_l_zca = Variable(torch.from_numpy(x_l_zca))
        x_u_rep_zca = Variable(torch.from_numpy(x_u_rep_zca))
        x_u_zca = Variable(torch.from_numpy(x_u_zca))
        if cuda:
            y_real, z_real, y = y_real.cuda(), z_real.cuda(), y.cuda()
            x_l_zca, x_u_rep_zca, x_u_zca = x_l_zca.cuda(), x_u_rep_zca.cuda(), x_u_zca.cuda()

        # classify input
        cla_out_y_l = classifier(x_l_zca, cuda=cuda)
        # calculate loss
        cla_cost_l = losses['ce'](cla_out_y_l, y)  # size_average in pytorch is by default

        # classify input for target
        cla_out_y_rep = classifier(x_u_rep_zca, cuda=cuda)
        target = cla_out_y_rep.detach()
        del cla_out_y_rep

        # classify input
        cla_out_y = classifier(x_u_zca, cuda=cuda)
        # calculate loss
        cla_cost_u = unsup_weight * losses['mse'](cla_out_y, target)

        y_m = y_real
        z_m = z_real
        gen_out_x_m = generator(z=z_m, y=y_m)
        gen_out_x_m_zca = whitener.apply(gen_out_x_m)

        # classify input
        cla_out_y_m = classifier(gen_out_x_m_zca, cuda=cuda)
        # calculate loss
        cla_cost_g = losses['ce'](cla_out_y_m, y_m) * w_g

        # sum individual losses for backward
        cla_cost = cla_cost_l + cla_cost_u + cla_cost_g

        cla_optimizer = optim.Adam(classifier.parameters(), betas=(b1_c, 0.999), lr=cla_lr)
        # zero the parameter gradients, optimize and update parameters
        cla_optimizer.zero_grad()
        cla_cost.backward()
        cla_optimizer.step()

        # update batch permutations
        if i_l == ((x_labelled.shape[0] // size_l) - 1):
            p_l = rng.permutation(x_labelled.shape[0])
            x_labelled = x_labelled[p_l]
            y_labelled = y_labelled[p_l]
        if i_u == (num_batches_u - 1):
            p_u = rng.permutation(x_unlabelled.shape[0]).astype('int32')

        running_cla_cost += cla_cost.cpu().numpy().mean()

    return running_cla_cost/epochs


def eval_classifier(num_batches_e, eval_x, eval_y, batch_size, whitener, classifier, cuda):

    accurracy = []
    for i in range(num_batches_e):
        x_eval = eval_x[i*batch_size:(i+1)*batch_size]
        y_eval = eval_y[i*batch_size:(i+1)*batch_size]
        x_eval_zca = whitener.apply(x_eval)
        x_eval_zca = Variable(torch.from_numpy(x_eval_zca))
        if cuda:
            x_eval_zca = x_eval_zca.cuda()

        cla_out_y_eval = classifier(x_eval_zca, cuda=cuda)

        accurracy_batch = accuracy_score(y_eval, cla_out_y_eval.cpu().data.numpy())
        accurracy.append(accurracy_batch)

    return np.mean(accurracy)


#
def train_gan(discriminator1, discriminator2, generator, inferentor, classifier, whitener,
              x_labelled, x_unlabelled, y_labelled, p_u_d, p_u_i,
              num_classes, batch_size, num_batches_u,
              batch_c, batch_l, batch_g,
              n_z, optimizers, losses, rng, cuda=False):

    '''

    Args:
        discriminator1(DConvNet1): Discriminator instance xy
        discriminator2(DConvNet2): Discriminator instance xz
        generator(Generator): Generator instance
        inferentor(InferenceNet): Inference Net instance
        classifier(ClassifierNet): Classifier Net instance
        whitener(ZCA): ZCA instance
        x_labelled: batch of labelled input data
        x_unlabelled: batch of unlabelled input data
        y_labelled: batch of corresponding labels
        p_u_d: data slice object (idx)
        p_u_i: data slice object (idx)
        num_classes(int): number of target classes
        batch_size(int): size of mini-batch
        num_batches_u:
        batch_c:
        batch_l:
        batch_g:
        n_z:
        optimizers(dict): dictionary containing optimizer instances for all respective nets (dis, gen, inf)
        losses(dict): dictionary containing respective loss instances (BCE, MSE, CE)
        b1: beta1 in Adam
        cuda(bool): cuda flag

    Returns:

    '''

    for i in range(num_batches_u):
            i_l = i % (x_labelled.shape[0] // batch_l)

            from_u_i = i*batch_size  # unlabelled inferentor slice
            to_u_i = (i+1)*batch_size
            from_u_d = i*batch_c    # unlabelled discriminator slice
            to_u_d = (i+1) * batch_c
            from_l = i_l*batch_l    # labelled
            to_l = (i_l+1)*batch_l

            # create samples and labels
            sample_y = torch.from_numpy(np.int32(np.repeat(np.arange(num_classes), int(batch_size/num_classes))))
            y_real = torch.from_numpy(np.int32(np.random.randint(10, size=batch_g)))
            z_real = torch.from_numpy(np.random.uniform(size=(batch_g, n_z)).astype(np.float32))
            z_rand = torch.rand((batch_size*n_z)).view(batch_size, n_z)

            sample_y, y_real, z_real, z_rand = Variable(sample_y), Variable(y_real), Variable(z_real), Variable(z_rand)
            if cuda:
                sample_y, y_real, z_real, z_rand = sample_y.cuda(), y_real.cuda(), z_real.cuda(), z_rand.cuda()

            dis_losses = train_discriminator(discriminator1=discriminator1,
                                             discriminator2=discriminator2,
                                             generator=generator,
                                             inferentor=inferentor,
                                             classificator=classifier,
                                             whitener=whitener,
                                             x_labelled=x_labelled[from_l:to_l],  # sym_x_l
                                             x_unlabelled=x_unlabelled,
                                             y_labelled=y_labelled[from_l:to_l],  # sym_y
                                             slice_x_dis=p_u_d[from_u_d:to_u_d],  # slice_x_u_d
                                             y_real=y_real,  # sym_y_m
                                             z_real=z_real,  # sym_z_m
                                             slice_x_inf=p_u_i[from_u_i:to_u_i],  # slice_x_u_i
                                             sample_y=sample_y,  # sym_y_g
                                             z_rand=z_rand,
                                             batch_size=batch_size,
                                             optimizer=optimizers['dis'],
                                             loss=losses['bce'],
                                             cuda=cuda)

            inf_losses = train_inferentor(x_unlabelled=x_unlabelled,
                                          sample_y=sample_y,
                                          generator=generator,
                                          z_rand=z_rand,
                                          discriminator2=discriminator2,
                                          inferentor=inferentor,
                                          mse=losses['mse'],
                                          bce=losses['bce'],
                                          slice_x_u_i=p_u_i[from_u_i:to_u_i],
                                          optimizer=optimizers['inf'],
                                          cuda=cuda)

            gen_losses = train_generator(whitener=whitener,
                                         optimizer=optimizers['gen'],
                                         BCE_loss=losses['bce'],
                                         MSE_loss=losses['mse'],
                                         cross_entropy_loss=losses['ce'],
                                         discriminator1=discriminator1,
                                         discriminator2=discriminator2,
                                         inferentor=inferentor,
                                         generator=generator,
                                         classifier=classifier,
                                         sample_y=sample_y,
                                         z_rand=z_rand)


            gan_loss = {
                'dis': dis_losses,
                'inf': inf_losses,
                'gen': gen_losses
            }

            if i_l == ((x_labelled.shape[0] // batch_l) - 1):
                p_l = rng.permutation(x_labelled.shape[0])
                x_labelled = x_labelled[p_l]
                y_labelled = y_labelled[p_l]



