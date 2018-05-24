#!/usr/bin/env python3

import numpy as np

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, BatchNormalization, Dense, Conv2DTranspose, Flatten, Reshape, \
    Lambda, LeakyReLU, Activation
from keras.regularizers import l2

from .losses import mean_gaussian_negative_log_likelihood


def create_models(recon_vs_gan_weight=1e-6, wdecay=1e-5, bn_mom=0.9, bn_eps=1e-6):

    image_shape = (64, 64, 3)
    n_channels = image_shape[-1]
    n_encoder = 1024
    n_discriminator = 512
    latent_dim = 128
    decode_from_shape = (8, 8, 256)
    n_decoder = np.prod(decode_from_shape)

    leaky_relu_alpha = 0.2

    def conv_block(x, filters, leaky=True, transpose=False):
        conv = Conv2DTranspose if transpose else Conv2D
        activation = LeakyReLU(leaky_relu_alpha) if leaky else Activation('relu')
        layers = [
            conv(filters, 5, strides=2, padding='same', kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform'),
            BatchNormalization(momentum=bn_mom, epsilon=bn_eps),
            activation
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
    y = Dense(n_encoder, kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = LeakyReLU(leaky_relu_alpha)(y)

    z_mean = Dense(latent_dim, name='z_mean', kernel_initializer='he_uniform')(y)
    z_log_var = Dense(latent_dim, name='z_log_var', kernel_initializer='he_uniform')(y)

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
        Dense(n_decoder, kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform', input_shape=(latent_dim,)),
        BatchNormalization(),
        LeakyReLU(leaky_relu_alpha),
        Reshape(decode_from_shape),
        *conv_block(None, 256, transpose=True),
        *conv_block(None, 128, transpose=True),
        *conv_block(None, 32, transpose=True),
        Conv2D(n_channels, 5, activation='tanh', padding='same', kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform', name='output_image')
    ], name='decoder')

    # Discriminator features at the 9th layer
    discriminator_features = Sequential([
        Conv2D(32, 5, padding='same', kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform', input_shape=image_shape),
        LeakyReLU(leaky_relu_alpha),
        *conv_block(None, 128, leaky=True),
        *conv_block(None, 256, leaky=True),
        conv_block(None, 256, leaky=True)[0]
    ], name='discriminator_features')

    # Complete discriminator model
    dis_input = Input(shape=image_shape, name='discriminator_input')
    d = discriminator_features(dis_input)
    d = BatchNormalization()(d)
    d = LeakyReLU(leaky_relu_alpha)(d)
    d = Flatten()(d)
    d = Dense(n_discriminator, kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform')(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(leaky_relu_alpha)(d)
    d = Dense(1, activation='sigmoid', kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform')(d)

    discriminator = Model(dis_input, d, name='discriminator')

    # Reconstructed output of VAE
    x_tilde = decoder(sampler(encoder.outputs))

    # Learned similarity metric
    dis_feat_tilde = discriminator_features(x_tilde)
    dis_feat = discriminator_features(x)
    dis_nll_loss = mean_gaussian_negative_log_likelihood(dis_feat, dis_feat_tilde)

    # KL divergence loss
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(kl_loss)

    encoder_train = Model(x, [dis_feat, dis_feat_tilde], name='encoder')
    encoder_train.add_loss(vae_loss)
    encoder_train.add_loss(dis_nll_loss)

    # z_p is sampled directly from isotropic gaussian
    z_p = Input(shape=(latent_dim,), name='z_p')
    x_p = decoder(z_p)

    dis_tilde = discriminator(x_tilde)
    dis_p = discriminator(x_p)

    decoder_train = Model([x, z_p], [dis_tilde, dis_p], name='decoder')
    normalized_weight = recon_vs_gan_weight / (1. - recon_vs_gan_weight)
    decoder_train.add_loss(normalized_weight * dis_nll_loss)

    vae = Model(x, x_tilde, name='vae')

    vaegan = Model(x, dis_tilde, name='vaegan')

    return encoder, decoder, discriminator, encoder_train, decoder_train, vae, vaegan
