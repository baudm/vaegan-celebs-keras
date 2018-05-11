#!/usr/bin/env python3

import numpy as np

from keras import Model
from keras.datasets import mnist
from keras.callbacks import TensorBoard

from vaegan.models import create_models

import cv2


def main():
    encoder, decoder, discriminator, vae, vae_loss = create_models()
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



    batch_size = 32

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


    # Assume images in x_train
    # x_train =  np.zeros((100, 64, 64, 3))

    discriminator.compile('rmsprop', 'binary_crossentropy', ['accuracy'])
    discriminator.trainable = False

    model = Model(vae.inputs, discriminator(vae.outputs), name='vaegan')
    model.add_loss(vae_loss)
    model.compile('rmsprop', 'binary_crossentropy', ['accuracy'])

    import keras.callbacks as cbks
    import os.path

    verbose = True
    checkpoint = cbks.ModelCheckpoint(os.path.join('.', 'model.{epoch:02d}.h5'), save_weights_only=True)

    callbacks = [TensorBoard(batch_size=batch_size), checkpoint]

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

    rand_indexes = np.random.randint(0, x_train.shape[0], size=1)
    real_images = x_train[rand_indexes]

    model.save_weights('trained.h5')

    a = encoder.predict(real_images)
    print(a)



if __name__ == '__main__':
    main()
