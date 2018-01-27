import torch
from torch.autograd import Variable
import numpy as np


def train_discriminator(discriminator1, discriminator2, generator, inferator, classificator, whitener,
                        x_labelled, x_unlabelled, y_labelled,
                        slice_x_dis, y_real, z_real, slice_x_inf, sample_y, z_rand,
                        batch_size, optimizer, loss, cuda):
    '''

    Args:
        discriminator1(DConvNet1): Discriminator instance xy
        discriminator2(DConvNet2): Discriminator instance xz
        generator(Generator): Generator instance
        inferator(Inference_Net): Inference Net instance
        classificator(Classifier_Net): Classifier Net instance
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
    inf_z = inferator(unlabel_inf)

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
    dis2_out_pg = discriminator2(gen_out_x, z_rand)

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

    dis1_cost = dis1_cost_p + dis1_cost_pg
    dis2_cost = dis2_cost_p + dis2_cost_pg

    total_cost = dis1_cost + dis2_cost

    # optimization routines and weight updates
    optimizer.zero_grad()
    total_cost.backward()
    optimizer.step()

    return [dis1_cost.cpu().numpy().mean(), dis2_cost.cpu().numpy().mean()]


def train_gan(discriminator1, discriminator2, generator, inferator, classificator, whitener,
              x_labelled, x_unlabelled, y_labelled, p_u_d, p_u_i,
              num_classes, batch_size, num_batches_u,
              batch_c, batch_l, batch_g,
              n_z, optimizers, losses, lr, cuda=False):

    '''

    Args:
        discriminator1(DConvNet1): Discriminator instance xy
        discriminator2(DConvNet2): Discriminator instance xz
        generator(Generator): Generator instance
        inferator(Inference_Net): Inference Net instance
        classificator(Classifier_Net): Classifier Net instance
        whitener(ZCA): ZCA instance
        x_labelled: batch of labelled input data
        x_unlabelled: batch of unlabelled input data
        y_labelled: batch of corresponding labels
        p_u_d:
        p_u_i:
        num_classes(int): number of target classes
        batch_size(int): size of mini-batch
        num_batches_u:
        batch_c:
        batch_l:
        batch_g:
        n_z:
        optimizers(dict): dictionary containing optimizer instances for all respective nets (dis, gen, inf)
        losses(dict): dictionary containing respective loss instances (BCE, MSE, CE)
        cuda(bool): cuda flag

    Returns:

    '''



    for i in range(num_batches_u):
            i_l = i % (x_labelled.shape[0] // batch_l)

            from_u_i = i*batch_size  # unlabelled inferator slice
            to_u_i = (i+1)*batch_size
            from_u_d = i*batch_c    # unlabelled discriminator slice
            to_u_d = (i+1) * batch_c
            from_l = i_l*batch_l    # labelled
            to_l = (i_l+1)*batch_l

            # create samples and labels
            sample_y = torch.from_numpy(np.int32(np.repeat(np.arange(num_classes), batch_size/num_classes)))
            y_real = torch.from_numpy(np.int32(np.random.randint(10, size=batch_g)))
            z_real = torch.from_numpy(np.random.uniform(size=(batch_g, n_z)).astype(np.float32))
            z_rand = torch.rand(sizes=(batch_size, n_z))

            sample_y, y_real, z_real, z_rand = Variable(sample_y), Variable(y_real), Variable(z_real), Variable(z_rand)
            if cuda:
                sample_y, y_real, z_real, z_rand = sample_y.cuda(), y_real.cuda(), z_real.cuda(), z_rand.cuda()

            dis_losses = train_discriminator(discriminator1=discriminator1,
                                             discriminator2=discriminator2,
                                             generator=generator,
                                             inferator=inferator,
                                             classificator=classificator,
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


