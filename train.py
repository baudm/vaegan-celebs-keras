#!/usr/bin/env python3

import os
import sys

import matplotlib.pyplot as plt

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import RMSprop

from vaegan.models import create_models
from vaegan.training import fit_models
from vaegan.data import celeba_loader, encoder_loader, decoder_loader, discriminator_loader, discriminator_loader_fake, NUM_SAMPLES, mnist_loader
from vaegan.callbacks import DecoderOutputGenerator


def main():
    encoder, decoder, discriminator, encoder_train, decoder_train, discriminator_train, vae, vaegan = create_models()

    vaegan.summary()

    if len(sys.argv) == 3:
        vaegan.load_weights(sys.argv[1])
        initial_epoch = int(sys.argv[2])
    else:
        initial_epoch = 0

    batch_size = 64

    rmsprop = RMSprop(lr=0.0003)

    encoder.trainable = False
    decoder.trainable = False
    discriminator_train.name = 'di'
    discriminator_train.compile(rmsprop, ['binary_crossentropy'] * 3, ['acc'] * 3)

    discriminator.trainable = False
    decoder.trainable = True
    decoder_train.name = 'de'
    decoder_train.compile(rmsprop, ['binary_crossentropy'] * 2, ['acc'] * 2)

    decoder.trainable = False
    encoder.trainable = True
    encoder_train.name = 'e'
    encoder_train.compile(rmsprop)


    checkpoint = ModelCheckpoint(os.path.join('.', 'model.{epoch:02d}.h5'), save_weights_only=True)
    decoder_sampler = DecoderOutputGenerator()

    callbacks = [checkpoint, decoder_sampler, TensorBoard()]

    epochs = 100

    steps_per_epoch = NUM_SAMPLES // batch_size

    img_loader = celeba_loader(batch_size, num_child=3)
    dis_loader = discriminator_loader(img_loader)
    dec_loader = decoder_loader(img_loader)
    enc_loader = encoder_loader(img_loader)

    models = [discriminator_train, decoder_train, encoder_train]
    generators = [dis_loader, dec_loader, enc_loader]
    metrics = [{'di_loss': 0, 'di_acc': 1}, {'de_loss': 0, 'de_acc': 3, 'de_acc_p': 5}, {'en_loss': 0}]

    fit_models(vaegan, models, generators, metrics, batch_size,
               steps_per_epoch=steps_per_epoch, callbacks=callbacks, epochs=epochs, initial_epoch=initial_epoch)

    vaegan.save_weights('trained.h5')

    x = next(celeba_loader(1))

    x_tilde = vae.predict(x)

    plt.subplot(211)
    plt.imshow((x[0].squeeze() + 1.) / 2., cmap='gray')

    plt.subplot(212)
    plt.imshow((x_tilde[0].squeeze() + 1.) / 2., cmap='gray')

    plt.show()


if __name__ == '__main__':
    main()
