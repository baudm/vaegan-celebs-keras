#!/usr/bin/env python3
import warnings

import numpy as np

from keras import Model
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from keras.utils import Sequence, OrderedEnqueuer, GeneratorEnqueuer

from vaegan.models import create_models

import cv2

from keras import callbacks as cbks

def fit_generator(models,
                  generators,
                  steps_per_epoch=None,
                  epochs=1,
                  verbose=1,
                  callbacks=None,
                  validation_data=None,
                  validation_steps=None,
                  class_weight=None,
                  max_queue_size=10,
                  workers=0,
                  use_multiprocessing=False,
                  shuffle=True,
                  initial_epoch=0):
    """Trains the model on data generated batch-by-batch by a Python generator or an instance of `Sequence`.

    The generator is run in parallel to the model, for efficiency.
    For instance, this allows you to do real-time data augmentation
    on images on CPU in parallel to training your model on GPU.

    The use of `keras.utils.Sequence` guarantees the ordering
    and guarantees the single use of every input per epoch when
    using `use_multiprocessing=True`.

    # Arguments
        generator: A generator or an instance of `Sequence`
            (`keras.utils.Sequence`) object in order to avoid
            duplicate data when using multiprocessing.
            The output of the generator must be either
            - a tuple `(inputs, targets)`
            - a tuple `(inputs, targets, sample_weights)`.
            This tuple (a single output of the generator) makes a single
            batch. Therefore, all arrays in this tuple must have the same
            length (equal to the size of this batch). Different batches may
            have different sizes. For example, the last batch of the epoch
            is commonly smaller than the others, if the size of the dataset
            is not divisible by the batch size.
            The generator is expected to loop over its data
            indefinitely. An epoch finishes when `steps_per_epoch`
            batches have been seen by the model.
        steps_per_epoch: Integer.
            Total number of steps (batches of samples)
            to yield from `generator` before declaring one epoch
            finished and starting the next epoch. It should typically
            be equal to the number of samples of your dataset
            divided by the batch size.
            Optional for `Sequence`: if unspecified, will use
            the `len(generator)` as a number of steps.
        epochs: Integer. Number of epochs to train the model.
            An epoch is an iteration over the entire data provided,
            as defined by `steps_per_epoch`.
            Note that in conjunction with `initial_epoch`,
            `epochs` is to be understood as "final epoch".
            The model is not trained for a number of iterations
            given by `epochs`, but merely until the epoch
            of index `epochs` is reached.
        verbose: Integer. 0, 1, or 2. Verbosity mode.
            0 = silent, 1 = progress bar, 2 = one line per epoch.
        callbacks: List of `keras.callbacks.Callback` instances.
            List of callbacks to apply during training.
            See [callbacks](/callbacks).
        validation_data: This can be either
            - a generator for the validation data
            - tuple `(x_val, y_val)`
            - tuple `(x_val, y_val, val_sample_weights)`
            on which to evaluate
            the loss and any model metrics at the end of each epoch.
            The model will not be trained on this data.
        validation_steps: Only relevant if `validation_data`
            is a generator. Total number of steps (batches of samples)
            to yield from `validation_data` generator before stopping
            at the end of every epoch. It should typically
            be equal to the number of samples of your
            validation dataset divided by the batch size.
            Optional for `Sequence`: if unspecified, will use
            the `len(validation_data)` as a number of steps.
        class_weight: Optional dictionary mapping class indices (integers)
            to a weight (float) value, used for weighting the loss function
            (during training only). This can be useful to tell the model to
            "pay more attention" to samples from an under-represented class.
        max_queue_size: Integer. Maximum size for the generator queue.
            If unspecified, `max_queue_size` will default to 10.
        workers: Integer. Maximum number of processes to spin up
            when using process-based threading.
            If unspecified, `workers` will default to 1. If 0, will
            execute the generator on the main thread.
        use_multiprocessing: Boolean.
            If `True`, use process-based threading.
            If unspecified, `use_multiprocessing` will default to `False`.
            Note that because this implementation relies on multiprocessing,
            you should not pass non-picklable arguments to the generator
            as they can't be passed easily to children processes.
        shuffle: Boolean. Whether to shuffle the order of the batches at
            the beginning of each epoch. Only used with instances
            of `Sequence` (`keras.utils.Sequence`).
            Has no effect when `steps_per_epoch` is not `None`.
        initial_epoch: Integer.
            Epoch at which to start training
            (useful for resuming a previous training run).

    # Returns
        A `History` object. Its `History.history` attribute is
        a record of training loss values and metrics values
        at successive epochs, as well as validation loss values
        and validation metrics values (if applicable).

    # Raises
        ValueError: In case the generator yields data in an invalid format.

    # Example

    ```python
        def generate_arrays_from_file(path):
            while True:
                with open(path) as f:
                    for line in f:
                        # create numpy arrays of input data
                        # and labels, from each line in the file
                        x1, x2, y = process_line(line)
                        yield ({'input_1': x1, 'input_2': x2}, {'output': y})

        model.fit_generator(generate_arrays_from_file('/my_file.txt'),
                            steps_per_epoch=10000, epochs=10)
    ```
    """
    wait_time = 0.01  # in seconds
    epoch = initial_epoch

    do_validation = bool(validation_data)

    is_sequence = isinstance(generators, Sequence)
    if not is_sequence and use_multiprocessing and workers > 1:
        warnings.warn(
            UserWarning('Using a generator with `use_multiprocessing=True`'
                        ' and multiple workers may duplicate your data.'
                        ' Please consider using the`keras.utils.Sequence'
                        ' class.'))
    if steps_per_epoch is None:
        if is_sequence:
            steps_per_epoch = len(generators)
        else:
            raise ValueError('`steps_per_epoch=None` is only valid for a'
                             ' generator based on the `keras.utils.Sequence`'
                             ' class. Please specify `steps_per_epoch` or use'
                             ' the `keras.utils.Sequence` class.')

    # python 2 has 'next', 3 has '__next__'
    # avoid any explicit version checks
    val_gen = (hasattr(validation_data, 'next') or
               hasattr(validation_data, '__next__') or
               isinstance(validation_data, Sequence))
    if (val_gen and not isinstance(validation_data, Sequence) and
            not validation_steps):
        raise ValueError('`validation_steps=None` is only valid for a'
                         ' generator based on the `keras.utils.Sequence`'
                         ' class. Please specify `validation_steps` or use'
                         ' the `keras.utils.Sequence` class.')

    # Prepare display labels.
    out_labels = {}
    labels = []
    for model in models:
        print(model.name, model.metrics_names)
        out_labels[model.name] = model.metrics_names
        labels.extend([model.name + '_' + n for n in model.metrics_names])

    callback_metrics = labels + ['val_' + n for n in labels]

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
    callback_model = models[-1]
    for model in models:
        if hasattr(model, 'callback_model') and model.callback_model:
            callback_model = model.callback_model
    callbacks.set_model(callback_model)
    callbacks.set_params({
        'epochs': epochs,
        'steps': steps_per_epoch,
        'verbose': verbose,
        'do_validation': do_validation,
        'metrics': callback_metrics,
    })
    callbacks.on_train_begin()

    enqueuer = None
    val_enqueuer = None

    try:
        # if do_validation and not val_gen:
        #     # Prepare data for validation
        #     if len(validation_data) == 2:
        #         val_x, val_y = validation_data
        #         val_sample_weight = None
        #     elif len(validation_data) == 3:
        #         val_x, val_y, val_sample_weight = validation_data
        #     else:
        #         raise ValueError('`validation_data` should be a tuple '
        #                          '`(val_x, val_y, val_sample_weight)` '
        #                          'or `(val_x, val_y)`. Found: ' +
        #                          str(validation_data))
        #     val_x, val_y, val_sample_weights = models._standardize_user_data(
        #         val_x, val_y, val_sample_weight)
        #     val_data = val_x + val_y + val_sample_weights
        #     if models.uses_learning_phase and not isinstance(K.learning_phase(), int):
        #         val_data += [0.]
        #     for cbk in callbacks:
        #         cbk.validation_data = val_data


        output_generators = []
        for generator in generators:
            if workers > 0:
                if is_sequence:
                    enqueuer = OrderedEnqueuer(generator,
                                               use_multiprocessing=use_multiprocessing,
                                               shuffle=shuffle)
                else:
                    enqueuer = GeneratorEnqueuer(generator,
                                                 use_multiprocessing=use_multiprocessing,
                                                 wait_time=wait_time)
                enqueuer.start(workers=workers, max_queue_size=max_queue_size)
                output_generator = enqueuer.get()
            else:
                if is_sequence:
                    output_generator = iter(generator)
                else:
                    output_generator = generator

            output_generators.append(output_generator)

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
                # if x is None or len(x) == 0:
                #     # Handle data tensors support when no input given
                #     # step-size = 1 for data tensors
                #     batch_size = 1
                # elif isinstance(x, list):
                #     batch_size = x[0].shape[0]
                # elif isinstance(x, dict):
                #     batch_size = list(x.values())[0].shape[0]
                # else:
                #     batch_size = x.shape[0]
                batch_size = 64
                batch_logs['batch'] = batch_index
                batch_logs['size'] = batch_size
                callbacks.on_batch_begin(batch_index, batch_logs)

                for model, output_generator in zip(models, output_generators):

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

                    outs = model.train_on_batch(x, y,
                                                 sample_weight=sample_weight,
                                                 class_weight=class_weight)

                    if not isinstance(outs, list):
                        outs = [outs]


                    # print(model.name, outs)de [8.032661, 3.8838384, 4.1488247, 0.375, 0.375, 0.515625, 0.515625]

                    if model.name is 'de':
                        batch_logs['de_di_acc1'] = outs[3]
                        batch_logs['de_di_acc2'] = outs[5]
                    else:
                        labels = out_labels[model.name]
                        for l, o in zip(labels, outs):
                            batch_logs[model.name + '_' + l] = o



                callbacks.on_batch_end(batch_index, batch_logs)

                batch_index += 1
                steps_done += 1

                # Epoch finished.
                # if steps_done >= steps_per_epoch and do_validation:
                #     if val_gen:
                #         val_outs = models.evaluate_generator(
                #             validation_data,
                #             validation_steps,
                #             workers=workers,
                #             use_multiprocessing=use_multiprocessing,
                #             max_queue_size=max_queue_size)
                #     else:
                #         # No need for try/except because
                #         # data has already been validated.
                #         val_outs = models.evaluate(
                #             val_x, val_y,
                #             batch_size=batch_size,
                #             sample_weight=val_sample_weights,
                #             verbose=0)
                #     if not isinstance(val_outs, list):
                #         val_outs = [val_outs]
                #     # Same labels assumed.
                #     for l, o in zip(out_labels, val_outs):
                #         epoch_logs['val_' + l] = o

                if callback_model.stop_training:
                    break

            callbacks.on_epoch_end(epoch, epoch_logs)
            epoch += 1
            if callback_model.stop_training:
                break

    finally:
        try:
            if enqueuer is not None:
                enqueuer.stop()
        finally:
            if val_enqueuer is not None:
                val_enqueuer.stop()

    callbacks.on_train_end()
    return models[-1].history


def main():
    encoder, decoder, discriminator, encoder_train, decoder_train, vae, vaegan = create_models()



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
    #
    # vae.compile('rmsprop', 'mse')
    # vae.load_weights('vae2.h5')
    # x = x_train[0].reshape(1, 64, 64, 1)
    # p = vae.predict(x)
    # import matplotlib.pyplot as plt
    #
    # plt.subplot(211)
    # plt.imshow(x.squeeze(), cmap='gray')
    # plt.subplot(212)
    # plt.imshow(p.squeeze(), cmap='gray')
    # plt.show()
    # return
    # vae.fit(x_train, x_train, epochs=2, batch_size=32)
    # vae.save_weights('vae2.h5')
    # return


    # Assume images in x_train
    # x_train =  np.zeros((100, 64, 64, 3))

    from keras.optimizers import SGD, RMSprop
    sgd = SGD(nesterov=True)

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
    decoder_train.compile(rmsprop, 2*['binary_crossentropy'], 2*['acc'])
    decoder_train.metrics_names = ['l', 'di_l', 'di_acc1', 'di_acc2']

    # model = decoder_train


    #
    # model = Model(vae.inputs, discriminator(vae.outputs), name='vaegan')
    # model.add_loss(vae.losses)
    # model.summary()
    #
    # # model.load_weights('model.04.h5')
    #
    # model.compile('adam', 'binary_crossentropy', ['accuracy'])


    vaegan.compile('adam','mse')



    import keras.callbacks as cbks
    import os.path

    verbose = True
    checkpoint = cbks.ModelCheckpoint(os.path.join('.', 'model.{epoch:02d}.h5'), save_weights_only=True)


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

    fit_generator([discriminator, decoder_train, encoder_train], [disc_gen(), dec_gen(), enc_gen()],
                  steps_per_epoch=steps_per_epoch, callbacks=callbacks, epochs=epochs
                  )

    #
    # do_validation = False
    #
    # callback_metrics = ['e_l', 'de_l', 'acc_xh', 'acc_xp', 'di_l', 'di_acc']
    #
    # model.history = cbks.History()
    # callbacks = [cbks.BaseLogger()] + (callbacks or []) + [model.history]
    # if verbose:
    #     callbacks += [cbks.ProgbarLogger(count_mode='steps')]
    # callbacks = cbks.CallbackList(callbacks)
    #
    # # it's possible to callback a different model than self:
    # if hasattr(model, 'callback_model') and model.callback_model:
    #     callback_model = model.callback_model
    # else:
    #     callback_model = model
    # callbacks.set_model(callback_model)
    # callbacks.set_params({
    #     'epochs': epochs,
    #     'steps': steps_per_epoch,
    #     'verbose': verbose,
    #     'do_validation': do_validation,
    #     'metrics': callback_metrics,
    # })
    # callbacks.on_train_begin()
    #
    # epoch_logs = {}
    #
    # import matplotlib.pyplot as plt
    #
    # for epoch in range(epochs):
    #
    #     callbacks.on_epoch_begin(epoch)
    #
    #     for batch_index in range(steps_per_epoch):
    #         batch_logs = {}
    #         batch_logs['batch'] = batch_index
    #         batch_logs['size'] = batch_size
    #         callbacks.on_batch_begin(batch_index, batch_logs)
    #
    #
    #         rand_indexes = np.random.randint(0, x_train.shape[0], size=batch_size)
    #         x = x_train[rand_indexes]
    #         # real_images = x_train[batch_index*batch_size:(batch_index+1)*batch_size]#rand_indexes]
    #
    #         x_hat = vae.predict(x)
    #
    #
    #         # print(fake_images.shape)
    #         half_batch = batch_size // 2
    #
    #         z_p = np.random.normal(size=(batch_size, 128))
    #         x_p = decoder.predict(z_p)
    #
    #         inputs = np.concatenate([x, x_hat, x_p])
    #
    #         # Label real and fake images
    #         y = np.zeros([3 * batch_size, 1], dtype='float32')
    #         y[:batch_size, :] = 1
    #
    #         #
    #         # # Train the Discriminator network
    #         # if batch_index % 2:
    #         #     inputs = x
    #         #     y = np.ones([batch_size, 1], dtype='float32')
    #         # else:
    #         #     inputs = x_hat
    #         #     y = np.zeros([batch_size, 1], dtype='float32')
    #
    #         metrics = discriminator.train_on_batch(inputs, y)
    #         batch_logs['di_l'] = metrics[0]
    #         batch_logs['di_acc'] = metrics[1]
    #         # print('discriminator', metrics)
    #
    #         y_real = np.ones([batch_size, 1], dtype='float32')
    #         y_fake = np.zeros([batch_size, 1], dtype='float32')
    #
    #
    #         metrics = decoder_train.train_on_batch([x, z_p], [y_real, y_fake, y_fake])
    #         batch_logs['de_l'] = metrics[0]
    #         batch_logs['acc_xh'] = metrics[2]
    #         batch_logs['acc_xp'] = metrics[3]
    #
    #         metrics = encoder_train.train_on_batch(x, None)
    #         batch_logs['e_l'] = metrics[0]
    #
    #         # rand_indexes = np.random.randint(0, x_train.shape[0], size=batch_size)
    #         # x = x_train[rand_indexes]
    #         # y = np.ones([batch_size, 1], dtype='float32')
    #         # vg_metrics = model.train_on_batch(x, y)
    #         # print('full', metrics)
    #         #
    #         #
    #         # batch_logs['vaegan_loss'] = vg_metrics[0]
    #         # batch_logs['vaegan_accuracy'] = vg_metrics[1]
    #
    #         callbacks.on_batch_end(batch_index, batch_logs)
    #
    #
    #     callbacks.on_epoch_end(epoch, batch_logs)

    rand_indexes = np.random.randint(0, x_train.shape[0], size=1)
    x = x_train[rand_indexes]

    vaegan.save_weights('trained.h5')

    import matplotlib.pyplot as plt

    x_hat = vae.predict(x)

    plt.subplot(211)
    plt.imshow((x[0].squeeze() + 1.) / 2., cmap='gray')

    plt.subplot(212)
    plt.imshow((x_hat[0].squeeze() + 1.) / 2., cmap='gray')

    plt.show()


if __name__ == '__main__':
    main()
