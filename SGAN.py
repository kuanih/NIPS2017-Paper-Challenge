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
    #                                  tile_shape=(num_classes, num_labelled/num_classes),
    #                                  colorImg=colorImg, scale=generation_scale,
    #                                 save_path=os.path.join(outfolder, 'x_l_'+str(ssl_data_seed)+'_sgan.png'))

num_batches_l = x_labelled.shape[0] / batch_size
num_batches_u = x_unlabelled.shape[0] / batch_size
num_batches_e = eval_x.shape[0] / batch_size_eval

print('num_batches_l: ', num_batches_l)
print('num_batches_u: ', num_batches_u)
print('num_batches_e: ', num_batches_e)