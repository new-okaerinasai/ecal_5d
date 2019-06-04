import os
import tensorflow as tf
import numpy as np
from tqdm import trange
from datetime import datetime
from tensorflow.python.client import device_lib


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


BATCH_SIZE = 128
Z_SIZE = 16
INPUT_SIZE = 30 * 30
AUX_DIM = 5
N_ITERS = 14000
PROPOSAL_TYPE = "conv"

import matplotlib.pyplot as plt


def show_imgs(imgs, name, ncol=10, nrow=10):
    plt.figure(figsize=(20, 16))
    for i in range(ncol * nrow):
        plt.subplot(nrow, ncol, i + 1)
        plt.imshow(imgs[i])
    plt.savefig(name + ".pdf")


with tf.variable_scope("dataset"):
    responses = tf.placeholder(dtype=tf.float32,
                               shape=(None, 30, 30, 1),
                               name="train_images")
    aux = tf.placeholder(dtype=tf.float32,
                         shape=(None, AUX_DIM),
                         name="train_labels")
    responses_dataset = tf.data.Dataset.from_tensor_slices([responses, ])
    responses_dataset = responses_dataset.batch(BATCH_SIZE)
    responses_iterator = responses_dataset.make_initializable_iterator()
    X = responses  # responses_iterator.get_next()
    # X = tf.reshape(X, (BATCH_SIZE, 30, 30, 1))

    aux_dataset = tf.data.Dataset.from_tensors([aux, ])
    aux_dataset = aux_dataset.batch(BATCH_SIZE)
    aux_iterator = aux_dataset.make_initializable_iterator()
    y = aux  # aux_iterator.get_next()
    # y = tf.reshape(y, (BATCH_SIZE, AUX_DIM))

    print(y.shape, X.shape)

with tf.variable_scope("CVAE"):
    with tf.variable_scope("prior_network"):
        upsample1 = tf.layers.batch_normalization(tf.layers.dense(y, 512, activation=tf.nn.relu))
        upsample2 = tf.layers.batch_normalization(tf.layers.dense(upsample1, 128, activation=tf.nn.relu))
        upsample3 = tf.layers.batch_normalization(tf.layers.dense(upsample2, 64, activation=tf.nn.relu))

        prior_mu = tf.layers.dense(upsample3, Z_SIZE)
        prior_sigma = tf.layers.dense(upsample3, Z_SIZE, activation=tf.nn.softplus)

        with tf.variable_scope("sampling"):
            eps = tf.random_normal((BATCH_SIZE, Z_SIZE))
            cond_embedding = prior_mu + prior_sigma * eps

    with tf.variable_scope("proposal_network"):
        X_flatten = tf.reshape(X, (-1, 30 * 30 * 1))
        if PROPOSAL_TYPE == "conv":
            y_feature_map = tf.reshape(tf.layers.dense(y, 30 * 30 * 1), (-1, 30, 30, 1))
            X_input = tf.concat([X, y_feature_map], axis=-1)
            x = tf.layers.conv2d(X_input, 128, (3, 3), activation=tf.nn.relu)
            x = tf.layers.max_pooling2d(x, (2, 2), strides=(2, 2))
            x = tf.layers.batch_normalization(x)

            x = tf.layers.conv2d(x, 64, (3, 3), activation=tf.nn.relu)
            x = tf.layers.max_pooling2d(x, (2, 2), strides=(2, 2))
            x = tf.layers.batch_normalization(x)

            x = tf.layers.conv2d(x, 32, (2, 2), activation=tf.nn.relu)
            x = tf.layers.max_pooling2d(x, (2, 2), strides=(2, 2))
            x = tf.layers.batch_normalization(x)

            x = tf.layers.flatten(x)
        else:
            X_input = tf.concat([X_flatten, y], axis=1)
            x = tf.layers.batch_normalization(tf.layers.dense(X_input, 512, activation=tf.nn.relu))
            x = tf.layers.batch_normalization(tf.layers.dense(x, 256, activation=tf.nn.relu))
            x = tf.layers.batch_normalization(tf.layers.dense(x, 64, activation=tf.nn.relu))

        proposal_mu = tf.layers.dense(x, Z_SIZE)
        proposal_sigma = tf.layers.dense(x, Z_SIZE, activation=tf.nn.softplus)

        with tf.variable_scope("sampling"):
            eps = tf.random_normal((BATCH_SIZE, Z_SIZE))
            z = proposal_mu + proposal_sigma * eps

    with tf.variable_scope("generative_network"):
        bn1 = tf.layers.BatchNormalization()
        fc1 = tf.layers.Dense(64, activation=tf.nn.relu)

        bn2 = tf.layers.BatchNormalization()
        fc2 = tf.layers.Dense(128, activation=tf.nn.relu)

        bn3 = tf.layers.BatchNormalization()
        fc3 = tf.layers.Dense(512, activation=tf.nn.relu)

        downsample1 = bn1(fc1(z)) + upsample3
        downsample2 = bn2(fc2(downsample1)) + upsample2
        downsample3 = bn3(fc3(downsample2)) + upsample1

        output = tf.concat([downsample3, y], axis=-1)

        fc_mu = tf.layers.Dense(30*30*1)
        fc_sigma = tf.layers.Dense(30*30*1, activation=tf.nn.softplus)

        out_mu = fc_mu(output)
        out_sigma = fc_sigma(output)

        with tf.variable_scope("sampling"):
            eps = tf.random_normal((BATCH_SIZE, 900))
            generated_images = tf.reshape(out_mu + out_sigma * eps, (-1, 30, 30, 1))

    with tf.variable_scope("generator"):
        z_test = tf.random_normal(shape=(BATCH_SIZE, Z_SIZE))
        downsample1 = bn1(fc1(z_test)) + upsample3
        downsample2 = bn2(fc2(downsample1)) + upsample2
        downsample3 = bn3(fc3(downsample2)) + upsample1

        output = tf.concat([downsample3, y], axis=-1)
        out_mu_test = fc_mu(output)
        out_sigma_test = fc_sigma(output)

        with tf.variable_scope("sampling"):
            eps = tf.random_normal((BATCH_SIZE, 900))
            test_images = tf.reshape(out_mu_test + out_sigma_test * eps, (-1, 30, 30, 1))


with tf.variable_scope("loss"):
    with tf.variable_scope("kl"):
        mean_square = tf.square(proposal_mu - prior_mu)

        kl_loss = tf.log(prior_sigma + 1e-8) - tf.log(proposal_sigma + 1e-8) - 0.5
        kl_loss += (proposal_sigma ** 2 + mean_square) / (2 * prior_sigma ** 2 + 1e-8)

    with tf.variable_scope("reconstruction_loss"):
        X_diff = X_flatten - out_mu
        X_power = -0.5 * tf.square(X_diff) / (out_sigma ** 2 + 1e-8)
        reconstruction_loss = (tf.log(out_sigma ** 2 + 1e-8)) + 2 * tf.log(2 * np.pi) / 2. - X_power
        # out_logvar + 2 * tf.log(2 * np.pi)) / 2. - X_power

    loss = tf.reduce_sum(kl_loss) + tf.reduce_sum(reconstruction_loss)

global_step = tf.Variable(0, trainable=False)

with tf.variable_scope("optimizer"):
    learning_rate = 1e-4
    learning_step = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
    ).minimize(loss,
               global_step=global_step
               )


path = "./real_data.npz"

real_data = np.load(path)
data_img = real_data["EnergyDeposit"][..., None]
n_data_img = []

min_ = np.min(data_img)
max_ = np.max(data_img)

for i, elem in enumerate(data_img):
    if (i % 5000 == 0):
        print (i)
    n_elem = (elem) / (max_ - min_)
    n_data_img.append(n_elem)
data_img = (data_img - data_img.mean()) / data_img.std()
print(data_img.min(), data_img.max())
n_data_img = data_img[:-BATCH_SIZE]  # np.array(n_data_img)
test_data_img = data_img[-BATCH_SIZE:]

data_point = real_data['ParticlePoint']
data_momentum = real_data['ParticleMomentum']
data_x = np.concatenate([data_point[:, :2], data_momentum], axis=1)

data_x = (data_x - data_x.mean()) / data_x.std()

data_x = data_x[:-BATCH_SIZE]
data_x_test = data_x[-BATCH_SIZE:]

# summaries for tensorboard
tf.summary.scalar("loss/loss", loss)
tf.summary.scalar("loss/kl_loss", tf.reduce_sum(kl_loss))
tf.summary.scalar("loss/reconstruction_loss", tf.reduce_sum(reconstruction_loss))

tf.summary.image("imges/generated_images", generated_images[:25], max_outputs=25)
tf.summary.image("images/real_images", n_data_img[:25], max_outputs=25)

merged = tf.summary.merge_all()


path = "./logs_" + str(datetime.now())

writer = tf.summary.FileWriter(path)

# saver

path = "./snapshot_iter_{}".format(0)

# train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    sess.run([responses_iterator.initializer, aux_iterator.initializer],
             {responses: n_data_img, aux: data_x})
    n_batches = int(len(n_data_img) // BATCH_SIZE)

    for epoch in trange(N_ITERS):
        for batch_ in range(n_batches - 1):
            summary, _ = sess.run([merged, learning_step],
                                  {responses: n_data_img[batch_ * BATCH_SIZE:(batch_ + 1) * BATCH_SIZE],
                                   y: data_x[batch_ * BATCH_SIZE:(batch_ + 1) * BATCH_SIZE]})
            writer.add_summary(summary, sess.run(global_step))
        writer.flush()
        print(sess.run([tf.reduce_mean(kl_loss), tf.reduce_mean(reconstruction_loss)],
                       {responses: n_data_img[batch_ * BATCH_SIZE:(batch_ + 1) * BATCH_SIZE],
                        y: data_x[batch_ * BATCH_SIZE:(batch_ + 1) * BATCH_SIZE]}
                       ))
        if (epoch) % 10 == 0:
            saver.save(sess, path)
	generated_ = sess.run([test_images],
			      {y: data_x_test[:128]})[0][..., 0]
	generated_[generated_ < 0] = 0
	show_imgs(generated_, "generated")
	show_imgs(test_data_img[:128, :, :, 0], "real")

