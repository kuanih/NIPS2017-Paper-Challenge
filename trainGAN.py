import torch
from torch.autograd import Variable
import numpy as np
import time



def train_discriminator(x_labelled, y_labelled, p_u_d, y_real, z_real, p_u_i, sample_y, optimizer, lr):






    dis_out_p = ll.get_output(dis_layers[-1],
                              {dis_in_x: T.concatenate([sym_x_l, sym_x_u_d, gen_out_x_m], axis=0)[:batch_size],
                               dis_in_y: T.concatenate([sym_y, cla_out_y_d_hard, sym_y_m], axis=0)[:batch_size]})
    dis_out_p_g = ll.get_output(dis_layers[-1], {dis_in_x: gen_out_x, dis_in_y: sym_y_g}, deterministic=False)

    disxz_out_p = ll.get_output(disxz_layers[-1], {disxz_in_x: sym_x_u_i, disxz_in_z: inf_z}, deterministic=False)
    disxz_out_p_g = ll.get_output(disxz_layers[-1], {disxz_in_x: gen_out_x, disxz_in_z: sym_z_rand},
                                  deterministic=False)






def train_gan(discriminator1, discriminator2, generator, inferator, whitener,
              x_labelled, y_labelled, p_u_d, p_u_i,
              num_classes, batch_size, num_batches_u,
              batch_c, batch_l, batch_g,
              n_z, optimizer, lr, cuda=False):


    for i in range(num_batches_u):
            i_l = i % (x_labelled.shape[0] // batch_l)

            from_u_i = i*batch_size
            to_u_i = (i+1)*batch_size
            from_u_d = i*batch_c
            to_u_d = (i+1) * batch_c
            from_l = i_l*batch_l
            to_l = (i_l+1)*batch_l

            sample_y = torch.from_numpy(np.int32(np.repeat(np.arange(num_classes), batch_size/num_classes)))
            y_real = torch.from_numpy(np.int32(np.random.randint(10, size=batch_g)))
            z_real = torch.from_numpy(np.random.uniform(size=(batch_g, n_z)).astype(np.float32))

            sample_y, y_real, z_real = Variable(sample_y), Variable(y_real), Variable(z_real)
            if cuda:
                sample_y, y_real, z_real = sample_y.cuda(), y_real.cuda(), z_real.cuda()



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


