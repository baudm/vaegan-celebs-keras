#!/usr/bin/env python3

import os

import numpy as np

import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import RMSprop

from vaegan.models import create_models

import cv2

from keras import callbacks as cbks

def fit_generator(callback_model,
                  models,
                  generators,
                  metrics_names,
                  batch_size,
                  steps_per_epoch=None,
                  epochs=1,
                  verbose=1,
                  callbacks=None,
                  initial_epoch=0):
    epoch = initial_epoch

    # Prepare display labels.
    callback_metrics = [n for m in metrics_names for n in m.keys()]

    # prepare callbacks
    stateful_metric_names = []
    for model in models:
        model.history = cbks.History()
        try:
            stateful_metric_names.extend(model.stateful_metric_names)
        except AttributeError:
            stateful_metric_names.extend(model.model.stateful_metric_names)
    _callbacks = [cbks.BaseLogger(
        stateful_metrics=stateful_metric_names)]
    if verbose:
        _callbacks.append(
            cbks.ProgbarLogger(
                count_mode='steps',
                stateful_metrics=stateful_metric_names))
    _callbacks += (callbacks or []) + [model.history for model in models]
    callbacks = cbks.CallbackList(_callbacks)

    # it's possible to callback a different model than self:
    callbacks.set_model(callback_model)
    callbacks.set_params({
        'epochs': epochs,
        'steps': steps_per_epoch,
        'verbose': verbose,
        'do_validation': False,
        'metrics': callback_metrics,
    })
    callbacks.on_train_begin()

    try:
        callback_model.stop_training = False
        # Construct epoch logs.
        epoch_logs = {}
        while epoch < epochs:
            for model in models:
                try:
                    stateful_metric_functions = model.stateful_metric_functions
                except AttributeError:
                    stateful_metric_functions = model.model.stateful_metric_functions
                for m in stateful_metric_functions:
                    m.reset_states()
            callbacks.on_epoch_begin(epoch)
            steps_done = 0
            batch_index = 0
            while steps_done < steps_per_epoch:

                # build batch logs
                batch_logs = {}
                batch_logs['batch'] = batch_index
                batch_logs['size'] = batch_size
                callbacks.on_batch_begin(batch_index, batch_logs)

                for model, output_generator, metrics in zip(models, generators, metrics_names):

                    generator_output = next(output_generator)

                    if not hasattr(generator_output, '__len__'):
                        raise ValueError('Output of generator should be '
                                         'a tuple `(x, y, sample_weight)` '
                                         'or `(x, y)`. Found: ' +
                                         str(generator_output))

                    if len(generator_output) == 2:
                        x, y = generator_output
                        sample_weight = None
                    elif len(generator_output) == 3:
                        x, y, sample_weight = generator_output
                    else:
                        raise ValueError('Output of generator should be '
                                         'a tuple `(x, y, sample_weight)` '
                                         'or `(x, y)`. Found: ' +
                                         str(generator_output))

                    outs = model.train_on_batch(x, y, sample_weight=sample_weight)

                    if not isinstance(outs, list):
                        outs = [outs]

                    for name, i in metrics.items():
                        batch_logs[name] = outs[i]

                callbacks.on_batch_end(batch_index, batch_logs)

                batch_index += 1
                steps_done += 1

                # Epoch finished.
                if callback_model.stop_training:
                    break

            callbacks.on_epoch_end(epoch, epoch_logs)
            epoch += 1
            if callback_model.stop_training:
                break

    finally:
        pass

    callbacks.on_train_end()
    return [model.history for model in models]


def main():
    encoder, decoder, discriminator, encoder_train, decoder_train, vae, vaegan = create_models()

    batch_size = 64

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Resize to 64x64
    x_train_new = np.zeros((x_train.shape[0], 64, 64), dtype='int32')
    for i, img in enumerate(x_train):
        x_train_new[i] = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)

    x_train = x_train_new
    del x_train_new

    # Normalize to [-1, 1]
    #x_train = np.pad(x_train, ((0, 0), (18, 18), (18, 18)), mode='constant', constant_values=0)
    x_train = np.expand_dims(x_train, -1)
    x_train = (x_train.astype('float32') - 127.5) / 127.5
    x_train = np.clip(x_train, -1., 1.)

    rmsprop = RMSprop(lr=0.0003)

    discriminator.name = 'di'
    discriminator.compile(rmsprop, 'binary_crossentropy', ['accuracy'])
    discriminator.trainable = False

    decoder.trainable = False
    encoder_train.name = 'e'
    encoder_train.compile(rmsprop)

    encoder.trainable = False
    decoder.trainable = True
    decoder_train.name = 'de'
    decoder_train.compile(rmsprop, 2 * ['binary_crossentropy'], 2 * ['acc'])


    checkpoint = ModelCheckpoint(os.path.join('.', 'model.{epoch:02d}.h5'), save_weights_only=True)

    callbacks = [checkpoint]





    epochs = 10
    steps_per_epoch = (x_train.shape[0]) // batch_size

    def disc_gen():
        half_batch = batch_size // 2
        quarter_batch = half_batch // 2

        mode = 0

        while True:
            rand_indexes = np.random.randint(0, x_train.shape[0], size=batch_size)
            x = x_train[rand_indexes]

            if mode == 0:
                inputs = x
                y = np.ones([batch_size, 1], dtype='float32')
            elif mode == 1:
                x_hat = vae.predict(x[half_batch:])
                z_p = np.random.normal(size=(half_batch, 128))
                x_p = decoder.predict(z_p)
                inputs = np.concatenate([x_hat, x_p])
                y = np.zeros([batch_size, 1], dtype='float32')
            #
            #
            # # real_images = x_train[batch_index*batch_size:(batch_index+1)*batch_size]#rand_indexes]
            #
            #
            # x_hat = vae.predict(x[quarter_batch:])
            #
            # # print(fake_images.shape)
            #
            # z_p = np.random.normal(size=(quarter_batch, 128))
            # x_p = decoder.predict(z_p)
            #
            # inputs = np.concatenate([x, x_hat, x_p])
            #
            # # Label real and fake images
            # y = np.zeros([batch_size, 1], dtype='float32')
            # y[:half_batch, :] = 1

            mode += 1
            if mode > 1:
                mode = 0

            yield inputs, y


    def dec_gen():
        while True:
            rand_indexes = np.random.randint(0, x_train.shape[0], size=batch_size)
            x = x_train[rand_indexes]
            z_p = np.random.normal(size=(batch_size, 128))
            y_real = np.ones([batch_size, 1], dtype='float32')
            yield [x, z_p], [y_real, y_real]

    def enc_gen():
        while True:
            rand_indexes = np.random.randint(0, x_train.shape[0], size=batch_size)
            x = x_train[rand_indexes]
            yield x, None

    fit_generator(vaegan, [discriminator, decoder_train, encoder_train], [disc_gen(), dec_gen(), enc_gen()],
                  [{'di_loss': 0, 'di_acc': 1}, {'de_loss': 0, 'de_acc': 3, 'de_acc_p': 5}, {'en_loss': 0}],
                  batch_size,
                  steps_per_epoch=steps_per_epoch, callbacks=callbacks, epochs=epochs
                  )

    rand_indexes = np.random.randint(0, x_train.shape[0], size=1)
    x = x_train[rand_indexes]

    vaegan.save_weights('trained.h5')

    x_hat = vae.predict(x)

    plt.subplot(211)
    plt.imshow((x[0].squeeze() + 1.) / 2., cmap='gray')

    plt.subplot(212)
    plt.imshow((x_hat[0].squeeze() + 1.) / 2., cmap='gray')

    plt.show()


if __name__ == '__main__':
    main()
