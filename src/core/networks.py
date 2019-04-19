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

from . import train
from . import costs
from .data import predict_with_K_fn
from .layer import stack_layers
from .util import LearningHandler, make_layer_list, train_gen, get_scale

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
            n_clusters, affinity, scale_nbr, n_nbrs, batch_sizes, normalized=True,
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
            W = costs.knn_affinity(input_affinity, n_nbrs, scale=scale, scale_nbr=scale_nbr)

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
        K.get_session().run(tf.variables_initializer(self.net.trainable_weights))

    def train(self, x_train_unlabeled, x_train_labeled, x_val_unlabeled,
            lr, drop, patience, num_epochs):
        # create handler for early stopping and learning rate scheduling
        self.lh = LearningHandler(
                lr=lr,
                drop=drop,
                lr_tensor=self.learning_rate,
                patience=patience)

        losses = np.empty((num_epochs,))
        val_losses = np.empty((num_epochs,))

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
                    batches_per_epoch=100)[0]

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

class SpectralVAEGAN:
    def __init__(self, inputs, spectralnet, orig_dim=2, latent_dim=1):
        from keras.layers import Input, Dense
        self.inputs = inputs['Unlabeled']
        # self.inputs = Input(shape=(latent_dim,))
        self.orig_dim = orig_dim
        self.latent_dim = latent_dim

        # make all spectralnet and siamese net layers untrainable
        x = [self.inputs]
        layers = []
        for l in spectralnet.net.layers[1:-1]:
            w = l.get_weights()
            n, m = w[0].shape
            new_l = Dense(m, activation='relu', input_shape=(n,), weights=w)
            new_l.trainable = False
            x.append(new_l(x[-1]))
            layers.append(new_l)

        pre_x = x[-1]
        # add orthonorm layer
        sess = K.get_session()
        # print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=''))
        with tf.variable_scope('', reuse=True):
            print("WE'RE HERE")
            v = tf.get_variable("ortho_weights_store")
        ows = sess.run(v)
        # n, m = ows.shape
        # b = np.ones((n,))
        # l = Dense(m, activation=None, input_shape=(n,), weights=(ows, b))
        t_ows = K.variable(ows)
        l = Lambda(lambda x: K.dot(x, t_ows))
        l.trainable = False
        x.append(l(x[-1]))
        layers.append(l)

        sn = Model(inputs=self.inputs, outputs=x[-2])
        import pickle
        with open('ows.pkl', 'wb') as f:
            pickle.dump(ows, f)
        sn.save('sn_fixed.h5')

        self.mu = mu = x = x[-1]

        # create encoder (only for sigma)
        x_enc = Dense(64, activation='relu')(pre_x)
        # x_enc = Dense(64, activation='relu')(self.inputs)
        # x_enc = Dense(64, activation='relu')(x_enc)
        # x_enc = Dense(256, activation='relu')(x_enc)

        # option 1
        self.sigma = sigma = Dense(latent_dim, activation='relu')(x_enc)
        self.mu = mu = Dense(latent_dim, activation='sigmoid')(x_enc)
        self.encoder = Model(self.inputs, [mu, sigma])
        # # option 2
        # self.sigma = sigma = self.x_enc = Dense(latent_dim, activation='sigmoid')(x_enc)
        # self.encoder = Model(self.inputs, sigma)

        # create decoder
        ls = []
        ls.append(Lambda(self.sampling, output_shape=(latent_dim,), name='z'))
        # ls.append(Dense(64, activation='relu'))
        ls.append(Dense(64, activation='relu'))
        # ls.append(Dense(256, activation='relu'))
        # ls.append(Dense(orig_dim, activation='relu'))
        ls.append(Dense(orig_dim, activation='sigmoid'))

        self.dec_layers = ls
        self.z = x_dec = ls[0]([mu, sigma])
        for l in ls[1:]:
            x_dec = l(x_dec)

        self.x_dec = x_dec

        self.decoder = Model(self.inputs, [self.x_dec])

        # # create discriminator
        # x_d = Dense(64, activation='relu')(x_dec)
        # x_d = Dense(64, activation='relu')(x_d)
        # x_d = Dense(256, activation='relu')(x_d)
        # self.x_d = Dense(1, activation='relu')(x_d)
        # self.discriminator = Model(self.inputs, x_d)

        # define loss
        # def d_loss(y_true, y_pred):
            # return K.mean(y_true * y_pred)

        from keras.losses import mse
        def vae_loss(y_true, y_pred):
            # reconstruction_loss = mse(self.inputs, self.x_dec) * self.orig_dim
            # reconstruction_loss = mse(self.inputs, y_pred) * self.orig_dim
            # reconstruction_loss = mse(y_true, y_pred) * self.orig_dim
            reconstruction_loss = tf.reduce_mean(tf.square(y_true - y_pred)) * self.orig_dim
            # kl_loss = 1 + sigma - K.square(mu) - K.exp(sigma)
            kl_loss = 1 + sigma - K.exp(sigma)
            kl_loss = K.sum(kl_loss, axis=-1)
            kl_loss *= -0.5
            kl_loss = 0
            # kl_loss *= 1/10
            vae_loss = K.mean(reconstruction_loss + kl_loss)
            return vae_loss

        self.vae = Model(inputs=self.inputs, outputs=self.x_dec)
        # vae_loss = vae_loss(None, None)
        # self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer='adam', loss=vae_loss)
        # self.vae.compile(optimizer='adam', loss=[vae_loss])

        # compile models
        # self.discriminator.compile(optimizer='adam', loss=[d_loss])

        # self.decoder.compile(optimizer='adam')

        # output_is_fake = self.discriminator(self.decoder(self.encoder(inputs=self.inputs)))

        # self.DGE = Model(inputs=self.inputs, outputs=output_is_fake)

        # self.DGE.compile(optimizer='adam', loss=[d_loss])

    def sampling(self, args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def generate_from_samples(self, x):
        # do we actually want to use sigma here, or draw from a distribution? you probably shouldn't in a VAE, but since we're already using the data point the generate a new one, maybe it does make sense.
        get_distribution = K.function([self.inputs], [self.mu, self.sigma])
        x_distr = predict_with_K_fn(get_distribution, x)
        x_recon = self.decoder.predict(x)
        return x_recon, x_distr

    def generate(self, n_samples=1):
        latent_dim = self.latent_dim
        mu = Input(shape=(latent_dim,))
        sigma = Input(shape=(latent_dim,))
        z = x_dec = self.dec_layers[0]([mu, sigma])
        for l in self.dec_layers[1:]:
            x_dec = l(x_dec)
        decoder = Model([mu, sigma], [x_dec])
        mus = np.zeros((n_samples, latent_dim))
        sigmas = np.ones((n_samples, latent_dim))
        k_z = K.get_session().run(z, feed_dict={mu: mus, sigma: sigmas})
        np.set_printoptions(suppress=True)
        print('Z', k_z)
        np.set_printoptions(suppress=None)
        x_recon = decoder.predict([mus, sigmas])
        return x_recon

    def train(self, xy_train, xy_val):
        x_train, y_train = xy_train
        x_val, y_val = xy_val
        earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
        batch_size = 1
        epochs = 10
        val_data = list((x_val, x_val))
        from keras.callbacks import Callback
        from keras.models import model_from_json
        import matplotlib.pyplot as plt
        class PlotExamples(Callback):
            def __init__(self):
                super(PlotExamples, self).__init__()
                json_path = '/home/henry/projects/SpectralVAEGAN/src/pretrain_weights/ae_mnist.json'
                weights_path = '/home/henry/projects/SpectralVAEGAN/src/pretrain_weights/ae_mnist_weights.h5'
                with open(json_path) as f:
                    self.pt_ae = pt_ae = model_from_json(f.read())
                pt_ae.load_weights(weights_path)

                self.get_reconstruction = K.function([pt_ae.layers[4].input], [pt_ae.output])
                self.targets = []  # collect y_true batches
                self.outputs = []  # collect y_pred batches

                # the shape of these 2 variables will change according to batch shape
                # to handle the "last batch", specify `validate_shape=False`
                # self.var_y_true = tf.Variable(0., validate_shape=False)
                # self.var_y_pred = tf.Variable(0., validate_shape=False)
                self.var_y_true = tf.Variable(0., validate_shape=False)
                self.var_y_pred = tf.Variable(0., validate_shape=False)

            def on_batch_end(self, batch, logs=None):
                y_true = K.eval(self.var_y_true)
                y_pred = K.eval(self.var_y_pred)
                self.targets.append(y_true)
                self.outputs.append(y_pred)
                print("BATCH", y_true.shape, y_pred.shape)
                true_img = predict_with_K_fn(self.get_reconstruction, y_true)[0]
                pred_img = predict_with_K_fn(self.get_reconstruction, y_pred)[0]
                fig = plt.figure()
                plt.subplot(1,2,1)
                plt.imshow(true_img.reshape((28, 28)))
                plt.subplot(1,2,2)
                plt.imshow(pred_img.reshape((28, 28)))
                loss = logs['loss']
                fig.suptitle('loss: {}'.format(loss))
                print("trying to show you graph")
                plt.show()


        pe = PlotExamples()

        fetches = [tf.assign(pe.var_y_true, self.vae.targets[0], validate_shape=False), tf.assign(pe.var_y_pred, self.vae.outputs[0], validate_shape=False)]
        self.vae._function_kwargs = {'fetches': fetches}

        self.vae.fit(x=x_train,
                y=x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=val_data,
                # callbacks=[earlystop, pe],
                callbacks=[earlystop],
                verbose=2)

