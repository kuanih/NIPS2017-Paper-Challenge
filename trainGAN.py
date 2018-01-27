import torch
from torch.autograd import Variable
import numpy as np


def train_discriminator(discriminator1, discriminator2, generator, inferator, classificator, whitener,
                        x_labelled, x_unlabelled, y_labelled,
                        slice_x_dis, y_real, z_real, slice_x_inf, sample_y, z_rand,
                        batch_size, d1_optimizer, d2_optimizer, loss, cuda):

    '''
    Parameter Translation
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

    # optimization routines and weight updates
    d1_optimizer.zero_grad()
    dis1_cost.backward()
    d1_optimizer.step()

    d2_optimizer.zero_grad()
    dis2_cost.backward()
    d2_optimizer.step()

    return [dis1_cost.cpu().numpy().mean(), dis2_cost.cpu().numpy().mean()]


def train_generator(optimizer, BCE_loss, MSE_loss, cross_entropy_loss,
                    loss_dis_generated, loss_dis_styled, loss_inference_z_g, n_z, 
                    classifier_out_y, z_rand, sample_y):
    '''
    Args:
        optimizer:          optimizer  for generator
        BCE_loss:           binary cross entropy loss
        MSE_loss:           mean squared error loss
        cross_entropy_loss: cross entropy loss
        loss_dis_generated: loss of discriminator 1
        loss_dis_styled:    loss of discriminator 2
        loss_inference_z_g: loss of inference
        n_z:                number of z
        classifier_out_y:   output of classifier
        z_rand:             random z sample
        sample_y:           sample of y

    Returns:

    '''
    # compute loss
    rz = MSE_loss(loss_inference_z_g, z_rand, n_z)
    ry = cross_entropy_loss(classifier_out_y, sample_y)

    gen_cost_p_g_1 = BCE_loss(loss_dis_generated, torch.ones(loss_dis_generated.shape))
    gen_cost_p_g_2 = BCE_loss(loss_dis_styled, torch.ones(loss_dis_styled.shape))

    generator_cost = gen_cost_p_g_1 + gen_cost_p_g_2 + rz + ry

    # optimization routines and weight updates
    optimizer.zero_grad()
    generator_cost.backward()
    optimizer.step()

    return generator_cost.cpu().numpy().mean()


def train_gan(discriminator1, discriminator2, generator, inferator, classificator, whitener,
              x_labelled, x_unlabelled, y_labelled, p_u_d, p_u_i,
              num_classes, batch_size, num_batches_u,
              batch_c, batch_l, batch_g,
              n_z, optimizers, losses, lr, cuda=False):

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
                                             d1_optimizer=optimizers['dis1'],
                                             d2_optimizer=optimizers['dis2'],
                                             loss=losses['bce'],
                                             cuda=cuda)

            # placeholder
            inference_loss = 0
            classificator_loss = 0

            generator_loss = train_generator(optimizer=optimizers['generator'],
                                             BCE_loss=losses['bce'],
                                             MSE_loss=losses['mse'],
                                             cross_entropy_loss=losses['ce'],
                                             loss_dis_generated=dis_losses[0],
                                             loss_dis_styled=dis_losses[1],
                                             loss_inference_z_g=inference_loss,
                                             n_z=n_z,
                                             classifier_out_y=classificator_loss,
                                             z_rand=z_rand,
                                             sample_y=sample_y)

            # for j in range(len(dl)):
            #    dl[j] += dl_b[j]

            # il_b = train_batch_inf(p_u_i[from_u_i:to_u_i], sample_y, lr)
            # for j in range(len(il)):
            #    il[j] += il_b[j]

            # gl_b = train_batch_gen(sample_y, lr)
            # for j in xrange(len(gl)):
            #    gl[j] += gl_b[j]

            # if i_l == ((x_labelled.shape[0] // batch_l) - 1):
            #    p_l = rng.permutation(x_labelled.shape[0])
            #    x_labelled = x_labelled[p_l]
            #    y_labelled = y_labelled[p_l]
