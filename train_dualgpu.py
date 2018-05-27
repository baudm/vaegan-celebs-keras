#!/usr/bin/env python3

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import RMSprop

from vaegan.models import create_models, build_graph
from vaegan.training import fit_models
from vaegan.data import celeba_loader, encoder_loader, decoder_loader, discriminator_loader, NUM_SAMPLES, mnist_loader
from vaegan.callbacks import DecoderOutputGenerator

from keras.utils import multi_gpu_model

class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)

def set_trainable(model, trainable):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


def main():
    encoder, decoder, discriminator = create_models()
    encoder_train, decoder_train, discriminator_train, vae, vaegan = build_graph(encoder, decoder, discriminator)

    if len(sys.argv) == 3:
        vaegan.load_weights(sys.argv[1])
        initial_epoch = int(sys.argv[2])
    else:
        initial_epoch = 0

    batch_size = 64

    rmsprop = RMSprop(lr=0.0003)

    set_trainable(encoder, False)
    set_trainable(decoder, False)
    discriminator_train.compile(rmsprop, ['binary_crossentropy'] * 3, ['acc'] * 3)
    discriminator_train.summary()

    set_trainable(discriminator, False)
    set_trainable(decoder, True)
    decoder_train.compile(rmsprop, ['binary_crossentropy'] * 2, ['acc'] * 2)
    decoder_train.summary()

    set_trainable(decoder, False)
    set_trainable(encoder, True)
    encoder_train.compile(rmsprop)
    encoder_train.summary()

    set_trainable(vaegan, True)

    checkpoint = ModelCheckpoint(os.path.join('.', 'model.{epoch:02d}.h5'), save_weights_only=True)
    decoder_sampler = DecoderOutputGenerator()

    callbacks = [checkpoint, decoder_sampler, TensorBoard()]

    epochs = 100

    steps_per_epoch = NUM_SAMPLES // batch_size

    seed = np.random.randint(2**32 - 1)

    img_loader = celeba_loader(batch_size, num_child=3, seed=seed)
    dis_loader = discriminator_loader(img_loader, seed=seed)
    dec_loader = decoder_loader(img_loader, seed=seed)
    enc_loader = encoder_loader(img_loader)

    models = [discriminator_train, decoder_train, encoder_train]
    generators = [dis_loader, dec_loader, enc_loader]
    metrics = [{'di_l': 1, 'di_l_t': 2, 'di_l_p': 3, 'di_a': 4, 'di_a_t': 7, 'di_a_p': 10}, {'de_l_t': 1, 'de_l_p': 2, 'de_a_t': 3, 'de_a_p': 5}, {'en_l': 0}]

    histories = fit_models(vaegan, models, generators, metrics, batch_size,
                           steps_per_epoch=steps_per_epoch, callbacks=callbacks,
                           epochs=epochs, initial_epoch=initial_epoch)

    vaegan.save_weights('trained.h5')

    x = next(celeba_loader(1))

    x_tilde = vae.predict(x)

    plt.subplot(211)
    plt.imshow((x[0].squeeze() + 1.) / 2.)

    plt.subplot(212)
    plt.imshow((x_tilde[0].squeeze() + 1.) / 2.)

    plt.show()


if __name__ == '__main__':
    main()
