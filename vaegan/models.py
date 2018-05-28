#!/usr/bin/env python3

import numpy as np

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, BatchNormalization, Dense, Conv2DTranspose, Flatten, Reshape, \
    Lambda, LeakyReLU, Activation
from keras.regularizers import l2

from .losses import mean_gaussian_negative_log_likelihood


def create_models(n_channels=3, recon_depth=9, wdecay=1e-5, bn_mom=0.9, bn_eps=1e-6):

    image_shape = (64, 64, n_channels)
    n_encoder = 1024
    n_discriminator = 512
    latent_dim = 128
    decode_from_shape = (8, 8, 256)
    n_decoder = np.prod(decode_from_shape)
    leaky_relu_alpha = 0.2

    def conv_block(x, filters, leaky=True, transpose=False, name=''):
        conv = Conv2DTranspose if transpose else Conv2D
        activation = LeakyReLU(leaky_relu_alpha) if leaky else Activation('relu')
        layers = [
            conv(filters, 5, strides=2, padding='same', kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform', name=name + 'conv'),
            BatchNormalization(momentum=bn_mom, epsilon=bn_eps, name=name + 'bn'),
            activation
        ]
        if x is None:
            return layers
        for layer in layers:
            x = layer(x)
        return x

    # Encoder
    def create_encoder():
        x = Input(shape=image_shape, name='enc_input')

        y = conv_block(x, 64, name='enc_blk_1_')
        y = conv_block(y, 128, name='enc_blk_2_')
        y = conv_block(y, 256, name='enc_blk_3_')
        y = Flatten()(y)
        y = Dense(n_encoder, kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform', name='enc_h_dense')(y)
        y = BatchNormalization(name='enc_h_bn')(y)
        y = LeakyReLU(leaky_relu_alpha)(y)

        z_mean = Dense(latent_dim, name='z_mean', kernel_initializer='he_uniform')(y)
        z_log_var = Dense(latent_dim, name='z_log_var', kernel_initializer='he_uniform')(y)

        return Model(x, [z_mean, z_log_var], name='encoder')

    # Decoder
    decoder = Sequential([
        Dense(n_decoder, kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform', input_shape=(latent_dim,), name='dec_h_dense'),
        BatchNormalization(name='dec_h_bn'),
        LeakyReLU(leaky_relu_alpha),
        Reshape(decode_from_shape),
        *conv_block(None, 256, transpose=True, name='dec_blk_1_'),
        *conv_block(None, 128, transpose=True, name='dec_blk_2_'),
        *conv_block(None, 32, transpose=True, name='dec_blk_3_'),
        Conv2D(n_channels, 5, activation='tanh', padding='same', kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform', name='dec_output')
    ], name='decoder')

    # Discriminator
    def create_discriminator():
        x = Input(shape=image_shape, name='dis_input')

        layers = [
            Conv2D(32, 5, padding='same', kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform', name='dis_blk_1_conv'),
            LeakyReLU(leaky_relu_alpha),
            *conv_block(None, 128, leaky=True, name='dis_blk_2_'),
            *conv_block(None, 256, leaky=True, name='dis_blk_3_'),
            *conv_block(None, 256, leaky=True, name='dis_blk_4_'),
            Flatten(),
            Dense(n_discriminator, kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform', name='dis_dense'),
            BatchNormalization(name='dis_bn'),
            LeakyReLU(leaky_relu_alpha),
            Dense(1, activation='sigmoid', kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform', name='dis_output')
        ]

        y = x
        y_feat = None
        for i, layer in enumerate(layers, 1):
            y = layer(y)
            # Output the features at the specified depth
            if i == recon_depth:
                y_feat = y

        return Model(x, [y, y_feat], name='discriminator')

    encoder = create_encoder()
    discriminator = create_discriminator()

    return encoder, decoder, discriminator


def _sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
       Instead of sampling from Q(z|X), sample eps = N(0,I)

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


def build_graph(encoder, decoder, discriminator, recon_vs_gan_weight=1e-6):
    image_shape = K.int_shape(encoder.input)[1:]
    latent_shape = K.int_shape(decoder.input)[1:]

    sampler = Lambda(_sampling, output_shape=latent_shape, name='sampler')

    # Inputs
    x = Input(shape=image_shape, name='input_image')
    # z_p is sampled directly from isotropic gaussian
    z_p = Input(shape=latent_shape, name='z_p')

    # Build computational graph

    z_mean, z_log_var = encoder(x)
    z = sampler([z_mean, z_log_var])

    x_tilde = decoder(z)
    x_p = decoder(z_p)

    dis_x, dis_feat = discriminator(x)
    dis_x_tilde, dis_feat_tilde = discriminator(x_tilde)
    dis_x_p = discriminator(x_p)[0]

    # Compute losses

    # Learned similarity metric
    dis_nll_loss = mean_gaussian_negative_log_likelihood(dis_feat, dis_feat_tilde)

    # KL divergence loss
    kl_loss = K.mean(-0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1))

    # Create models for training
    encoder_train = Model(x, dis_feat_tilde, name='e')
    encoder_train.add_loss(kl_loss)
    encoder_train.add_loss(dis_nll_loss)

    decoder_train = Model([x, z_p], [dis_x_tilde, dis_x_p], name='de')
    normalized_weight = recon_vs_gan_weight / (1. - recon_vs_gan_weight)
    decoder_train.add_loss(normalized_weight * dis_nll_loss)

    discriminator_train = Model([x, z_p], [dis_x, dis_x_tilde, dis_x_p], name='di')

    # Additional models for testing
    vae = Model(x, x_tilde, name='vae')
    vaegan = Model(x, dis_x_tilde, name='vaegan')

    return encoder_train, decoder_train, discriminator_train, vae, vaegan
