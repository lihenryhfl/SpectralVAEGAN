'''
networks.py: contains network definitions (for siamese net,
triplet siamese net, and spectralnet)
'''

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Lambda, Subtract
from keras.callbacks import EarlyStopping

from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Lambda, Subtract, Dense, Conv2DTranspose, Dropout, Reshape, Flatten, UpSampling2D, Activation
from keras.layers.merge import _Merge
from keras.activations import relu
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping
from keras.datasets import mnist, cifar10, fashion_mnist
from keras.losses import mse, binary_crossentropy

from functools import partial

import vdae.train as train
import vdae.costs as costs
from vdae.data import predict_with_K_fn
from vdae.layer import stack_layers
from vdae.util import LearningHandler, make_layer_list, train_gen, get_scale

class SiameseNet:
    def __init__(self, inputs, arch, siam_reg, y_true):
        self.orig_inputs = inputs
        # set up inputs
        self.inputs = {
                'A': inputs['Unlabeled'],
                'B': Input(shape=inputs['Unlabeled'].get_shape().as_list()[1:]),
                'Labeled': inputs['Labeled'],
                }

        self.y_true = y_true

        # generate layers
        self.layers = []
        self.layers += make_layer_list(arch, 'siamese', siam_reg)

        # create the siamese net
        self.outputs = stack_layers(self.inputs, self.layers)

        # add the distance layer
        self.distance = Lambda(costs.euclidean_distance, output_shape=costs.eucl_dist_output_shape)([self.outputs['A'], self.outputs['B']])

        #create the distance model for training
        self.net = Model([self.inputs['A'], self.inputs['B']], self.distance)

        # compile the siamese network
        self.net.compile(loss=costs.get_contrastive_loss(m_neg=1, m_pos=0.05), optimizer='rmsprop')

    def train(self, pairs_train, dist_train, pairs_val, dist_val,
            lr, drop, patience, num_epochs, batch_size):
        # create handler for early stopping and learning rate scheduling
        self.lh = LearningHandler(
                lr=lr,
                drop=drop,
                lr_tensor=self.net.optimizer.lr,
                patience=patience)

        # initialize the training generator
        train_gen_ = train_gen(pairs_train, dist_train, batch_size)

        # format the validation data for keras
        validation_data = ([pairs_val[:, 0], pairs_val[:, 1]], dist_val)

        # compute the steps per epoch
        steps_per_epoch = int(len(pairs_train) / batch_size)

        # train the network
        hist = self.net.fit_generator(train_gen_, epochs=num_epochs, validation_data=validation_data, steps_per_epoch=steps_per_epoch, callbacks=[self.lh])

        return hist

    def predict(self, x, batch_sizes):
        # compute the siamese embeddings of the input data
        return train.predict(self.outputs['A'], x_unlabeled=x, inputs=self.orig_inputs, y_true=self.y_true, batch_sizes=batch_sizes)

class SpectralNet:
    def __init__(self, inputs, arch, spec_reg, y_true, y_train_labeled_onehot,
            n_clusters, affinity, scale_nbr, n_nbrs, batch_sizes, normalized=False,
            siamese_net=None, x_train=None, have_labeled=False):
        self.y_true = y_true
        self.y_train_labeled_onehot = y_train_labeled_onehot
        self.inputs = inputs
        self.batch_sizes = batch_sizes
        self.normalized = normalized
        # generate layers
        self.layers = make_layer_list(arch[:-1], 'spectral', spec_reg)
        self.layers += [
                  {'type': 'tanh',
                   'size': n_clusters,
                   'l2_reg': spec_reg,
                   'name': 'spectral_{}'.format(len(arch)-1)},
                  {'type': 'Orthonorm', 'name':'orthonorm'}
                  ]

        # create spectralnet
        self.outputs = stack_layers(self.inputs, self.layers)
        self.net = Model(inputs=self.inputs['Unlabeled'], outputs=self.outputs['Unlabeled'])

        # DEFINE LOSS

        # generate affinity matrix W according to params
        if affinity == 'siamese':
            input_affinity = tf.concat([siamese_net.outputs['A'], siamese_net.outputs['Labeled']], axis=0)
            x_affinity = siamese_net.predict(x_train, batch_sizes)
        elif affinity in ['knn', 'full']:
            input_affinity = tf.concat([self.inputs['Unlabeled'], self.inputs['Labeled']], axis=0)
            x_affinity = x_train

        # calculate scale for affinity matrix
        scale = get_scale(x_affinity, self.batch_sizes['Unlabeled'], scale_nbr)

        # create affinity matrix
        if affinity == 'full':
            W = costs.full_affinity(input_affinity, scale=scale)
        elif affinity in ['knn', 'siamese']:
            W = costs.knn_affinity(input_affinity, n_nbrs, scale=scale, scale_nbr=scale_nbr, local_scale=True)

        # if we have labels, use them
        if have_labeled:
            # get true affinities (from labeled data)
            W_true = tf.cast(tf.equal(costs.squared_distance(y_true), 0),dtype='float32')

            # replace lower right corner of W with W_true
            unlabeled_end = tf.shape(self.inputs['Unlabeled'])[0]
            W_u = W[:unlabeled_end, :]                  # upper half
            W_ll = W[unlabeled_end:, :unlabeled_end]    # lower left
            W_l = tf.concat((W_ll, W_true), axis=1)      # lower half
            W = tf.concat((W_u, W_l), axis=0)

            # create pairwise batch distance matrix self.Dy
            y_ = tf.concat([self.outputs['Unlabeled'], self.outputs['Labeled']], axis=0)
        else:
            y_ = self.outputs['Unlabeled']

        if self.normalized:
            y_ = y_ / tf.reduce_sum(W, axis=1)

        self.Dy = costs.squared_distance(y_)

        # define loss
        self.loss = K.sum(W * self.Dy) / (2 * batch_sizes['Unlabeled'])

        # create the train step update
        self.learning_rate = tf.Variable(0., name='spectral_net_learning_rate')
        self.train_step = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss, var_list=self.net.trainable_weights)

        # initialize spectralnet variables
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            ortho_weights_store = tf.get_variable(name="ortho_weights_store")
        print("IMPORTANT:", self.net.trainable_weights, ortho_weights_store)
        # K.get_session().run(tf.variables_initializer(self.net.trainable_weights + [ortho_weights_store, self.learning_rate]))
        K.get_session().run(tf.global_variables_initializer())

    def train(self, x_train_unlabeled, x_train_labeled, x_val_unlabeled,
            lr, drop, patience, num_epochs, single_step=False):
        # create handler for early stopping and learning rate scheduling
        self.lh = LearningHandler(
                lr=lr,
                drop=drop,
                lr_tensor=self.learning_rate,
                patience=patience)

        losses = np.empty((num_epochs,))
        val_losses = np.empty((num_epochs,))

        bpe = 1 if single_step else 100

        # get validation loss
        val_loss_ = train.predict_sum(
                self.loss,
                x_unlabeled=x_val_unlabeled,
                inputs=self.inputs,
                y_true=self.y_true,
                x_labeled=x_train_unlabeled[0:0],
                y_labeled=self.y_train_labeled_onehot,
                batch_sizes=self.batch_sizes)
        print("Epoch: 0, val_loss={:2f}".format(val_loss_))

        # begin spectralnet training loop
        self.lh.on_train_begin()
        i = 0
        for i in range(num_epochs):
            # train spectralnet
            losses[i] = train.train_step(
                    return_var=[self.loss],
                    updates=self.net.updates + [self.train_step],
                    x_unlabeled=x_train_unlabeled,
                    inputs=self.inputs,
                    y_true=self.y_true,
                    batch_sizes=self.batch_sizes,
                    x_labeled=x_train_labeled,
                    y_labeled=self.y_train_labeled_onehot,
                    batches_per_epoch=bpe)[0]

            # get validation loss
            val_losses[i] = train.predict_sum(
                    self.loss,
                    x_unlabeled=x_val_unlabeled,
                    inputs=self.inputs,
                    y_true=self.y_true,
                    x_labeled=x_train_unlabeled[0:0],
                    y_labeled=self.y_train_labeled_onehot,
                    batch_sizes=self.batch_sizes)

            # do early stopping if necessary
            if self.lh.on_epoch_end(i, val_losses[i]):
                print('STOPPING EARLY')
                break

            # print training status
            print("Epoch: {}, loss={:2f}, val_loss={:2f}".format(i, losses[i], val_losses[i]))

        return losses[:i+1], val_losses[:i+1]

    def predict(self, x):
        # test inputs do not require the 'Labeled' input
        inputs_test = {'Unlabeled': self.inputs['Unlabeled'], 'Orthonorm': self.inputs['Orthonorm']}
        return train.predict(
                    self.outputs['Unlabeled'],
                    x_unlabeled=x,
                    inputs=inputs_test,
                    y_true=self.y_true,
                    x_labeled=x[0:0],
                    y_labeled=self.y_train_labeled_onehot[0:0],
                    batch_sizes=self.batch_sizes)

def pick_nearest_k(D, k, drop_self=True, randomize=False):
            n_batch = tf.shape(D)[-1]
            if drop_self:
                _, idxs = tf.nn.top_k(-D, k=k+1)
                idxs = idxs[:,1:]
            else:
                _, idxs = tf.nn.top_k(-D, k=k)
            # create a random index
            if randomize:
                range_D = tf.expand_dims(tf.range(n_batch * k), -1)
                p = tf.expand_dims(tf.random.uniform(shape=(n_batch * k,), maxval=k, dtype=tf.dtypes.int32), -1)
                p = tf.concat([range_D, p], axis=1)
                # draw from the top k values using this index
                idx = tf.expand_dims(tf.gather_nd(idxs, p), -1)
            else:
                print("IMPORTANT", idxs.shape)
                idx = tf.reshape(idxs, (n_batch * k, 1))

            return idx

class VDAE:
    def __init__(self, inputs, spectralnet, orig_dim,
                 alpha=1., normalize_factor=.1, k=16,
                 arch=[500], lr=1e-3):
        self.optimizer = optimizer = Adam(lr=lr)
        self.input = inputs['Unlabeled']
        self.orig_dim = orig_dim
        self.k = k
        self.arch = arch

        self.eps = 1e-5
        self.lam = 0.1
        self.alpha = 1.
        self.orig_dim = np.prod(self.orig_dim)

        self.x = x = self.copy_spectralnet(spectralnet)

        #
        # DEFINE LOSSES
        #
        def kl_loss(_, __):
            log_e = 2 * self.half_log_e
            v = tf.reshape(self.v, (-1, self.latent_dim, self.latent_dim))
            # find local neighborhood of each point in batch
            D = self.pairwise_distances(self.z, self.z_mu)
            n_batch = tf.shape(D)[-1]
            idx = pick_nearest_k(D, self.k, drop_self=True)
            omn = z_mu_nb = tf.gather_nd(self.z_mu, idx)
            z_mu_nb = tf.reshape(z_mu_nb, shape=(n_batch, self.k, self.latent_dim))
            print("z_mu_nb", z_mu_nb.shape)

            # obtain covariance of each local neighborhood
            cov_nb = tf.einsum('ikj,ikl->ijl', z_mu_nb, z_mu_nb)

            # obtain eigendecomposition of local neighborhood
            e_nb, v_nb = tf.linalg.eigh(cov_nb)
            e_nb += self.lam

            # take log (used in final loss) BEFORE truncating
            log_e_nb = tf.log(e_nb + self.eps)

            e_nb *= self.alpha

            # compute trace of Sigma_cov^{-1} Sigma_theta
            inv_sigma_nb = tf.einsum('ijk,ilk->ijl', tf.einsum('ijk,ik->ijk', v_nb, 1 / e_nb), v_nb)

            sigma_vae = tf.einsum('ijk,ilk->ijl', tf.einsum('ijk,ik->ijk', v, self.e), v)
            half_prod = tf.einsum('ikj,ikl->ijl', inv_sigma_nb, sigma_vae)
            trace = tf.linalg.svd(half_prod, compute_uv=False)

            # compute KL divergence
            self.kl_loss = tf.reduce_sum(log_e_nb - log_e - 1 + trace)
            return self.kl_loss
        def neighborhood_loss(_, __):
            # involves two bursts, on-manifold burst and off-manifold burst
            z = self.z
            x_recon = self.x_recon

            # obtain pairwise distances (size(recon) x size(input))
            D = self.pairwise_distances(z, self.z_mu)
            n_batch = tf.shape(D)[-1]
            idx = pick_nearest_k(D, self.k, drop_self=True)

            # now compute neighborhood
            orig_dim = np.prod(self.orig_dim)
            data_flat_shape = (-1, np.prod(self.orig_dim))
            input_ = tf.reshape(self.input, data_flat_shape)
            print("important input shape", input_.shape)
            input_neighborhood = tf.gather_nd(input_, idx)
            x_recon_flattened_expanded = tf.reshape(x_recon, data_flat_shape + (1,))
            x_recon = tf.reshape(tf.tile(x_recon_flattened_expanded, [1, self.k, 1]), data_flat_shape)
            print("NEIGHBORHOOD AND RECON SHAPES:", input_neighborhood.shape, x_recon.shape)
            self.neighborhood_loss = tf.reduce_sum(mse(input_neighborhood, x_recon)) / self.k

            return self.neighborhood_loss
        def vae_loss(_, __):
            return self.loss

        #
        # DEFINE LAYERS
        #

        # create encoder
        self.x_enc = x_enc = self.build_encoder(x, arch=self.arch)
        self.encoder = Model(inputs=self.input, outputs=x_enc)

        # create decoder
        self.x_recon = x_recon = self.build_decoder(x_enc, arch=self.arch)
        self.decoder_input = Input(shape=(self.latent_dim,), name='UnlabeledInput')
        self.decoder_output = self.build_decoder(self.decoder_input, arch=self.arch)
        self.decoder = Model(inputs=self.decoder_input, outputs=self.decoder_output)

        # create normalized decoder
        x_enc_norm = self.build_encoder(x, arch=self.arch, normalize_cov=normalize_factor)
        self.x_recon_norm = self.build_decoder(x_enc_norm, arch=self.arch)

        #
        # COMPUTE LOSS
        #
        losses = [kl_loss, neighborhood_loss]
        self.init_losses = [l(None, None) for l in losses]
        loss_weights = [1, 1]
        # initialize losses
        self.loss = sum([a * b if b != 0 else K.constant(0.) for a, b in zip(self.init_losses, loss_weights)])

        #
        # ASSEMBLE NETWORK
        #
        self.vae = Model(inputs=self.input, outputs=self.x_recon)
        self.vae.compile(optimizer=optimizer, loss=vae_loss)

    def pairwise_distances(self, A, B):
        r_A, r_B = tf.reduce_sum(A*A, 1), tf.reduce_sum(B*B, 1)

        # turn r into column vector
        r_A, r_B = tf.reshape(r_A, [-1, 1]), tf.reshape(r_B, [-1, 1])
        D = r_A - 2 * tf.matmul(A, B, transpose_b=True) + tf.transpose(r_B)

        return D

    def build_decoder(self, x, arch):
        if not hasattr(self, 'decoder_layers'):
            self.decoder_layers = [Dense(a, activation='relu') for a in arch]
            self.decoder_layers.append(Dense(self.orig_dim, activation='linear'))

        xs = [x]
        for l in self.decoder_layers:
            xs.append(l(xs[-1]))
        x = xs[-1]
        self.decoder_xs = xs
        return x

    def build_encoder(self, x, arch, normalize_cov=False, no_noise=False):

        if not hasattr(self, 'encoder_precov_layers'):
            self.encoder_precov_layers = [Dense(a, activation='relu') for a in arch]
            self.encoder_precov_layers.append(Dense(self.latent_dim * self.latent_dim, activation='linear'))
            self.encoder_eig_layers = [Dense(a, activation='relu') for a in arch]
            self.encoder_eig_layers.append(Dense(self.latent_dim, activation='linear'))

        # define mu (the latent embedding)
        mu = x
        if not hasattr(self, 'mu'):
            self.z_mu = mu

        x_precov = x
        # get covariance precursor
        for l in self.encoder_precov_layers:
            x_precov = l(x_precov)

        x_eig = x
        # get eigenvalues
        for l in self.encoder_eig_layers:
            x_eig = l(x_eig)

        # sample latent space (and normalize covariances if we're trying to do random walks)
        if not hasattr(self, 'encoder_sampling_layer'):
            f = partial(self.sampling, normalize_cov=normalize_cov)
            self.encoder_sampling_layer = Lambda(f, output_shape=(self.latent_dim,), name='z')

        if no_noise:
            cur_encoder_sampling_layer = Lambda(lambda x_: x_[0], output_shape=(self.latent_dim,))

        # get encoder embedding
        x_enc = self.encoder_sampling_layer([mu, x_precov, x_eig])

        return x_enc

    def copy_spectralnet(self, spectralnet):
        xs = [self.input]
        layers = []

        for l in spectralnet.net.layers[1:-1]:
            l.trainable = False
            xs.append(l(xs[-1]))
            layers.append(l)

        pre_x = xs[-1]
        # add orthonorm layer
        sess = K.get_session()
        with tf.variable_scope('', reuse=True):
            v = tf.get_variable("ortho_weights_store")
        ows = sess.run(v)
        t_ows = K.variable(ows)
        l = Lambda(lambda x: K.dot(x, t_ows))
        l.trainable = False
        xs.append(l(xs[-1]))
        layers.append(l)

        x = xs[-1]

        self.xs = xs

        self.sn = Model(inputs=self.input, outputs=x)

        self.spectralnet_dim = int(x.get_shape()[1])
        self.latent_dim = self.spectralnet_dim

        return x

    def sampling(self, args, normalize_cov):
        # get args
        z_mean, precov, e = args

        # reshape precov and compute cov = precov x precov.T
        cov = tf.reshape(precov, (-1, self.latent_dim, self.latent_dim))

        # perform eigendecomposition
        v, _ = tf.linalg.qr(cov)

        if not hasattr(self, 'e'):
            self.half_log_e, self.e, self.v = e, tf.exp(2 * e), tf.reshape(v, (-1, self.latent_dim * self.latent_dim))

        dim = self.latent_dim

        # get shapes
        batch = K.shape(z_mean)[0]

        # sample from normal distribution
        epsilon = K.random_normal(stddev=self.alpha, shape=(batch, K.int_shape(z_mean)[1]))

        # get sqrt covariance matrix stack
        sqrt_sigma = tf.einsum('ijk,ilk->ijl', tf.einsum('ijk,ik->ijk', v, tf.sqrt(self.e)), v)

        # multiply covariance matrix stack with random normal vector
        sqrt_sigma_epsilon = tf.einsum('ijk,ik->ij', sqrt_sigma, epsilon)

        if not hasattr(self, 'sqrt_sigma'):
            self.sqrt_sigma = tf.reshape(sqrt_sigma, (-1, self.latent_dim * self.latent_dim))

        # assembled output
        z = z_mean + sqrt_sigma_epsilon

        if not hasattr(self, 'z'):
            self.z = z

        return z

    def predict_from_spectralnet(self, x):
        get_fn = K.function([self.input], [self.x])
        return predict_with_K_fn(get_fn, x)

    def generate_from_samples(self, x, return_mu_sigma=False, normalize_cov=False):
        _x_recon = self.x_recon_norm if normalize_cov else self.x_recon
        get_fn = K.function([self.input], [_x_recon, self.z_mu, self.v, self.e, self.x_enc])
        x_recon, z_mu, z_sigma_v, z_sigma_lam, _x_enc = predict_with_K_fn(get_fn, x)
        if return_mu_sigma:
            return x_recon, z_mu, z_sigma_v, z_sigma_lam, _x_enc
        else:
            return x_recon

    def train(self, X_train, batch_size=128, epochs=100, full_batch=True):
        self.vae_loss = []
        last_cov = np.zeros((self.latent_dim, self.latent_dim))
        cov_x = X_train[np.random.randint(0, X_train.shape[0], 1)]
        for epoch in range(epochs):
            if full_batch:
                samples = X_train
            else:
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                samples = X_train[idx]

            vae_loss = self.vae.train_on_batch([samples], [samples])
            self.vae_loss.append(vae_loss)

            if epoch % 25 == 0:
                # plot the progress
                loss_names = ['kl_loss', 'neighborhood_loss']
                loss_string = "{} [VAE loss: {}] [" + ": {}] [".join(loss_names) + ": {}]"
                losses = self.init_losses
                loss_vals = K.get_session().run(losses, feed_dict={self.input: samples})
                print(loss_string.format(epoch, vae_loss, *loss_vals))

                # now get variance of the covariance vectors with respect to some fixed vector
                cov, val = K.get_session().run([self.v, self.e], feed_dict={self.input: cov_x})
                cov = cov.reshape((self.latent_dim, self.latent_dim))
                print('vector covariance:\n', cov.dot(last_cov.T))
                print(cov.T.dot(last_cov))
                print(val)
                last_cov = cov
