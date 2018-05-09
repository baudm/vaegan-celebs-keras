#!/usr/bin/env python3

import numpy as np

from keras.datasets import mnist
from keras.callbacks import TensorBoard
from vaegan.models import create_models


def main():
    encoder, decoder, discriminator, vae, model = create_models()
    #
    # encoder.compile('rmsprop', 'mse')
    #
    # x = np.random.uniform(-1.0, 1.0, size=[1, 64, 64, 1])
    # y1 = np.random.uniform(-1.0, 1.0, size=[1, 128])
    # y2 = np.random.uniform(-1.0, 1.0, size=[1, 128])
    #
    # encoder.fit(x, [y1, y2], callbacks=[TensorBoard()])
    #
    # return



    train_steps = 10000
    batch_size = 32

    (x_train, y_train), (x_test, y_test) = mnist.load_data()



    x_train = np.pad(x_train, ((0, 0), (18, 18), (18, 18)), mode='constant', constant_values=0)
    x_train = np.expand_dims(x_train, -1)
    x_train = (x_train.astype('float32') - 127.5) / 127.5
    x_train = np.clip(x_train, -1., 1.)


    # Assume images in x_train
    # x_train =  np.zeros((100, 64, 64, 3))

    discriminator.compile('rmsprop', 'binary_crossentropy', ['accuracy'])
    discriminator.trainable = False
    model.compile('rmsprop', 'binary_crossentropy', ['accuracy'])

    import keras.callbacks as cbks

    verbose = True
    callbacks = [TensorBoard(batch_size=batch_size)]

    epochs = 100
    steps_per_epoch = x_train.shape[0] // batch_size
    do_validation = False

    callback_metrics = ['disc_loss', 'disc_accuracy', 'vaegan_loss', 'vaegan_accuracy']

    model.history = cbks.History()
    callbacks = [cbks.BaseLogger()] + (callbacks or []) + [model.history]
    if verbose:
        callbacks += [cbks.ProgbarLogger(count_mode='steps')]
    callbacks = cbks.CallbackList(callbacks)

    # it's possible to callback a different model than self:
    if hasattr(model, 'callback_model') and model.callback_model:
        callback_model = model.callback_model
    else:
        callback_model = model
    callbacks.set_model(callback_model)
    callbacks.set_params({
        'epochs': epochs,
        'steps': steps_per_epoch,
        'verbose': verbose,
        'do_validation': do_validation,
        'metrics': callback_metrics,
    })
    callbacks.on_train_begin()

    epoch_logs = {}

    for epoch in range(epochs):

        callbacks.on_epoch_begin(epoch)

        for batch_index in range(steps_per_epoch):
            batch_logs = {}
            batch_logs['batch'] = batch_index
            batch_logs['size'] = batch_size
            callbacks.on_batch_begin(batch_index, batch_logs)


            rand_indexes = np.random.randint(0, x_train.shape[0], size=batch_size)
            real_images = x_train[rand_indexes]

            fake_images = vae.predict(real_images)
            # print(fake_images.shape)
            half_batch = batch_size // 2
            inputs = np.concatenate([real_images[:half_batch], fake_images[:half_batch]])

            # Label real and fake images
            y = np.ones([batch_size, 1], dtype='float32')
            y[half_batch:, :] = 0

            # Train the Discriminator network
            metrics = discriminator.train_on_batch(inputs, y)
            # print('discriminator', metrics)

            y = np.ones([batch_size, 1], dtype='float32')
            vg_metrics = model.train_on_batch(fake_images, y)
            # print('full', metrics)

            batch_logs['disc_loss'] = metrics[0]
            batch_logs['disc_accuracy'] = metrics[1]
            batch_logs['vaegan_loss'] = vg_metrics[0]
            batch_logs['vaegan_accuracy'] = vg_metrics[1]

            callbacks.on_batch_end(batch_index, batch_logs)

        callbacks.on_epoch_end(epoch, epoch_logs)


if __name__ == '__main__':
    main()
