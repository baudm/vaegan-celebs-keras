#!/usr/bin/env python3

import os.path
import glob
from multiprocessing.pool import ThreadPool as Pool

import numpy as np
from PIL import Image


NUM_SAMPLES = 202599

proj_root = os.path.split(os.path.dirname(__file__))[0]
images_path = os.path.join(proj_root, 'img_align_celeba_png', '*.png')


def _load_image(f):
    im = Image.open(f) \
              .crop((0, 20, 178, 198)) \
              .resize((64, 64), Image.BICUBIC)
    return np.asarray(im)


def celeba_loader(batch_size, normalize=True, num_child=4, seed=0, workers=8):
    rng = np.random.RandomState(seed)
    images = glob.glob(images_path)

    with Pool(workers) as p:
        while True:
            rng.shuffle(images)
            for s in range(0, len(images), batch_size):
                e = s + batch_size
                batch_names = images[s:e]
                batch_images = p.map(_load_image, batch_names)
                batch_images = np.stack(batch_images)

                if normalize:
                    batch_images = batch_images / 127.5 - 1.
                    # To be sure
                    batch_images = np.clip(batch_images, -1., 1.)

                # Yield the same batch num_child times since the images will be consumed
                # by num_child different child generators
                for i in range(num_child):
                    yield batch_images


def mnist_loader(batch_size, normalize=True, num_child=4, seed=0, workers=8):
    from keras.datasets import mnist
    (x_train, _), (_, _) = mnist.load_data()

    x_train_new = np.zeros((x_train.shape[0], 64, 64), dtype='int32')

    for i, img in enumerate(x_train):
        im = Image.fromarray(img).resize((64, 64), Image.BICUBIC)
        x_train_new[i] = np.asarray(im)

    x_train = x_train_new.reshape(-1, 64, 64, 1)
    del x_train_new

    if normalize:
        x_train = x_train / 127.5 - 1.
        # To be sure
        x_train = np.clip(x_train, -1., 1.)

    rng = np.random.RandomState(seed)
    while True:
        rng.shuffle(x_train)
        for s in range(0, len(x_train), batch_size):
            e = s + batch_size
            batch_images = x_train[s:e]

            # Yield the same batch num_child times since the images will be consumed
            # by num_child different child generators
            for i in range(num_child):
                yield batch_images


def discriminator_loader(img_loader):
    while True:
        x = next(img_loader)
        batch_size = x.shape[0]
        y = np.ones([batch_size, 1], dtype='float32')
        yield x, y


def discriminator_loader_fake(vae, decoder, img_loader, latent_dim=128, seed=0):
    rng = np.random.RandomState(seed)
    while True:
        x = next(img_loader)
        batch_size = x.shape[0]
        half_batch = batch_size // 2

        x_tilde = vae.predict(x[:half_batch])
        z_p = rng.normal(size=(half_batch, latent_dim))
        x_p = decoder.predict(z_p)
        inputs = np.concatenate([x_tilde, x_p])
        y = np.zeros([2 * half_batch, 1], dtype='float32')

        yield inputs, y


def decoder_loader(img_loader, latent_dim=128, seed=0):
    rng = np.random.RandomState(seed)
    while True:
        x = next(img_loader)
        batch_size = x.shape[0]
        # Sample z from isotropic Gaussian
        z_p = rng.normal(size=(batch_size, latent_dim))
        # Label as real
        y_real = np.ones([batch_size, 1], dtype='float32')
        yield [x, z_p], [y_real, y_real]


def encoder_loader(img_loader):
    while True:
        x = next(img_loader)
        yield x, None
