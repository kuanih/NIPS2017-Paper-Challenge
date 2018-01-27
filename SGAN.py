import time

import numpy as np
import utils

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
    #                                  tinference_losse_shape=(num_classes, num_labelled/num_classes),
    #                                  colorImg=colorImg, scale=generation_scale,
    #                                 save_path=os.path.join(outfolder, 'x_l_'+str(ssl_data_seed)+'_sgan.png'))

num_batches_l = x_labelled.shape[0] / batch_size
num_batches_unlabeled = x_unlabelled.shape[0] / batch_size
num_batches_e = eval_x.shape[0] / batch_size_eval

print('num_batches_l: ', num_batches_l)
print('num_batches_unlabeled: ', num_batches_unlabeled)
print('num_batches_e: ', num_batches_e)


def train_batch_discriminator(x_labelled, y_labelled, p_u_d, y_real, z_real,
                              p_u_i, sample_y, lr):
    return 0


def train_batch_generator(sample_y, lr):
    return 0


def train_batch_inference(p_u_i, sample_y, lr):
    return 0


print("x_labelled.shape[0] ", x_labelled.shape[0])
print(x_labelled.shape[0] // 50)


def discriminator_loss(discriminator_loss, from_l, to_l, p_u_d, from_u_d, to_u_d, y_real,
                        z_real, p_u_i, from_u_i, to_u_i, sample_y, lr):

    batch_discriminator_loss = train_batch_discriminator(x_labelled[from_l:to_l], y_labelled[from_l:to_l],
                                                         p_u_d[from_u_d:to_u_d], y_real, z_real,
                                                         p_u_i[from_u_i:to_u_i], sample_y, lr)

    for j in range(len(discriminator_loss)):
        discriminator_loss[j] += batch_discriminator_loss[j]

    for i in range(len(discriminator_loss)):
        discriminator_loss[i] /= num_batches_unlabeled


def inference_loss(p_u_i, from_u_i, to_u_i, sample_y, lr):

    batch_inference_loss = train_batch_inference(p_u_i[from_u_i:to_u_i], sample_y, lr)
    for j in range(len(inference_loss)):
        inference_loss[j] += batch_inference_loss[j]


def batch_generator_loss():
    return 0


def train_spec(batch_l, batch_c, batch_g, n_z, discriminator_loss, inference_loss, generator_loss,
               cl, lr, p_u_d, p_u_i, seed):

    # iterate over unlabeled data
    # num_batches_unlabeled = 250
    # x_labelled.shape[0] == 4000
    # batch_l = 50
    # 4000 // 50 = 80
    # if epoch > 500:  100 <= batch_c <= 140
    # if epoch > 500: batch_l = 50
    #
    for i in range(num_batches_unlabeled):
        i_l = i % (x_labelled.shape[0] // batch_l)      # 0 - 80 multiple times

        from_u_i = i * batch_size           # i * 200           range(0, 50000, 200)
        to_u_i = (i + 1) * batch_size       # (i + 1) * 200     range(1, 50200, 200)
        from_u_d = i * batch_c              # i * 120           range(0, 30000, 120)
        to_u_d = (i + 1) * batch_c          # (i + 1) * 120     range(1, 30120, 120)
        from_l = i_l * batch_l              # i_l * 50          range(0, 4000, 50) multiple times
        to_l = (i_l + 1) * batch_l          # (i_l + 1) * 50    range(1, 4050, 50) multiple times

        sample_y = np.int32(np.repeat(np.arange(num_classes), batch_size / num_classes))
        y_real = np.int32(np.random.randint(10, size=batch_g))
        z_real = np.random.uniform(size=(batch_g, n_z)).astype(np.float32)

        tmp = time.time()


        batch_generator_loss = train_batch_generator(sample_y, lr)
        for j in range(len(generator_loss)):
            generator_loss[j] += batch_generator_loss[j]

        if i_l == ((x_labelled.shape[0] // batch_l) - 1):
            rng = np.random.RandomState(seed)
            p_l = rng.permutation(x_labelled.shape[0])
            x_labelled = x_labelled[p_l]
            y_labelled = y_labelled[p_l]

    for i in range(len(generator_loss)):
        generator_loss[i] /= num_batches_unlabeled

    for i in range(len(cl)):
        cl[i] /= num_batches_unlabeled
