#!/usr/bin/env python3

import numpy as np

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Dense, Conv2DTranspose, Flatten, Reshape, \
    Lambda

from .losses import mean_gaussian_negative_log_likelihood


def create_models(wdecay=1e-5, bn_mom=0.9, bn_eps=1e-6, recon_depth=9, recon_vs_gan_weight=1e-6):
    image_shape = (64, 64, 1)
    n_channels = image_shape[-1]
    n_encoder = 1024
    n_discriminator = 512
    latent_dim = 128
    decode_from_shape = (8, 8, 256)
    n_decoder = np.prod(decode_from_shape)

    def conv_block(x, filters, transpose=False):
        conv = Conv2DTranspose if transpose else Conv2D
        layers = [
            conv(filters, 5, strides=2, padding='same'),
            BatchNormalization(),
            Activation('relu')
        ]
        if x is None:
            return layers
        for layer in layers:
            x = layer(x)
        return x

    # Encoder
    x = Input(shape=image_shape, name='input_image')

    y = conv_block(x, 64)
    y = conv_block(y, 128)
    y = conv_block(y, 256)
    y = Flatten()(y)
    y = Dense(n_encoder)(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    z_mean = Dense(latent_dim, name='z_mean')(y)
    z_log_var = Dense(latent_dim, name='z_log_var')(y)

    encoder = Model(x, [z_mean, z_log_var], name='encoder')

    # reparameterization trick
    # instead of sampling from Q(z|X), sample eps = N(0,I)
    # z = z_mean + sqrt(var)*eps
    def sampling(args):
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.
        # Arguments:
            args (tensor): mean and log of variance of Q(z|X)
        # Returns:
            z (tensor): sampled latent vector
        """

        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    sampler = Lambda(sampling, output_shape=(latent_dim,))

    # Decoder
    decoder = Sequential([
        Dense(n_decoder, input_shape=(latent_dim,)),
        BatchNormalization(),
        Activation('relu'),
        Reshape(decode_from_shape),
        *conv_block(None, 256, transpose=True),
        *conv_block(None, 128, transpose=True),
        *conv_block(None, 32, transpose=True),
        Conv2D(n_channels, 5, activation='tanh', padding='same', name='output_image')
    ], name='decoder')

    # Discriminator
    discriminator = Sequential([
        Conv2D(32, 5, padding='same', input_shape=image_shape),
        Activation('relu'),
        *conv_block(None, 128),
        *conv_block(None, 256),
        *conv_block(None, 256),
        Flatten(),
        Dense(n_discriminator),
        BatchNormalization(),
        Activation('relu'),
        Dense(1, activation='sigmoid')
    ], name='discriminator')

    discriminator_lth = Sequential(discriminator.layers[:recon_depth], name='discriminator_lth')


    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(kl_loss)



    x_hat = decoder(sampler(encoder.outputs))

    dis_lth_hat = discriminator_lth(x_hat)
    dis_lth = discriminator_lth(x)

    dist_like_loss = mean_gaussian_negative_log_likelihood(dis_lth, dis_lth_hat)

    encoder_train = Model(x, [dis_lth, dis_lth_hat], name='encoder')
    encoder_train.add_loss(vae_loss)
    encoder_train.add_loss(dist_like_loss)

    # batch_size = 32
    # mean = K.constant(0, dtype='float32', shape=(batch_size, latent_dim), name='mean')
    # var = K.constant(1, dtype='float32', shape=(batch_size, latent_dim), name='variance')

    z_p = Input(shape=(latent_dim,), name='z_p')
    x_p = decoder(z_p)

    dis_hat = discriminator(x_hat)
    dis_p = discriminator(x_p)

    decoder_train = Model([x, z_p], [dis_hat, dis_p], name='decoder')
    decoder_train.add_loss(recon_vs_gan_weight * dist_like_loss)


    vae = Model(x, x_hat, name='vae')
    #
    #
    vaegan = Model(x, dis_hat, name='vaegan')
    # vaegan.add_loss(vae_loss)


    return encoder, decoder, discriminator, encoder_train, decoder_train, vae, vaegan


# from keras.utils.vis_utils import plot_model
# e, d, dis, vae, gan = create_models()
# plot_model(e, show_shapes=True, to_file='encoder.png')
# plot_model(d, show_shapes=True, to_file='decoder.png')
# plot_model(dis, show_shapes=True, to_file='discriminator.png')
# plot_model(vae, show_shapes=True, to_file='vae.png')
# plot_model(gan, show_shapes=True, to_file=' gan.png')
