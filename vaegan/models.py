#!/usr/bin/env python3

import numpy as np

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Dense, Conv2DTranspose, Flatten, Reshape, \
    Lambda
from keras.regularizers import l2


def create_models(wdecay=1e-5, bn_mom=0.9, bn_eps=1e-6):
    image_shape = (64, 64, 1)
    n_channels = image_shape[-1]
    n_encoder = 1024
    n_discriminator = 512
    latent_dim = 128
    epsilon_std = 1.0
    decode_from_shape = (8, 8, 256)
    n_decoder = np.prod(decode_from_shape)
    l2_regularizer = l2(wdecay)

    def conv_block(x, filters, transpose=False):
        conv = Conv2DTranspose if transpose else Conv2D
        layers = [
            conv(filters, 5, strides=2, padding='same', kernel_regularizer=l2_regularizer),
            BatchNormalization(momentum=bn_mom, epsilon=bn_eps),
            Activation('relu')
        ]
        if x is None:
            return layers
        for layer in layers:
            x = layer(x)
        return x

    # Encoder
    input_image = Input(shape=image_shape, name='input_image')
    x = input_image
    for f in [64, 128, 256]:
        x = conv_block(x, f)
    x = Flatten()(x)
    x = Dense(n_encoder, kernel_regularizer=l2_regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    encoder = Model(input_image, [z_mean, z_log_var], name='encoder')

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # Decoder
    decoder = Sequential([
        Dense(n_decoder, kernel_regularizer=l2_regularizer, input_shape=(latent_dim,)),
        BatchNormalization(),
        Activation('relu'),
        Reshape(decode_from_shape),
        *conv_block(None, 256, transpose=True),
        *conv_block(None, 128, transpose=True),
        *conv_block(None, 32, transpose=True),
        Conv2D(n_channels, 5, activation='tanh', padding='same', kernel_regularizer=l2_regularizer, name='output_image')
    ], name='decoder')

    # Discriminator
    discriminator = Sequential([
        Conv2D(32, 5, activation='relu', padding='same', kernel_regularizer=l2_regularizer, input_shape=image_shape),
        *conv_block(None, 128),
        *conv_block(None, 256),
        *conv_block(None, 256),
        Flatten(),
        Dense(n_discriminator, kernel_regularizer=l2_regularizer),
        BatchNormalization(),
        Activation('relu'),
        Dense(1, activation='sigmoid', kernel_regularizer=l2_regularizer)
    ], name='discriminator')

    vae = Model(input_image, decoder(z), name='vae')

    z_sampled = Input(shape=(latent_dim,), name='z_sampled')
    gan = Model(z_sampled, discriminator(decoder(z_sampled)), name='gan')

    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(kl_loss)
    full_model = Model(input_image, discriminator(vae(input_image)), name='full_model')
    full_model.add_loss(vae_loss)

    return encoder, decoder, discriminator, vae, gan, full_model

#
# from keras.utils.vis_utils import plot_model
# e, d, dis, vae, gan = create_models()
# plot_model(e, show_shapes=True, to_file='encoder.png')
# plot_model(d, show_shapes=True, to_file='decoder.png')
# plot_model(dis, show_shapes=True, to_file='discriminator.png')
# plot_model(vae, show_shapes=True, to_file='vae.png')
# plot_model(gan, show_shapes=True, to_file='gan.png')
