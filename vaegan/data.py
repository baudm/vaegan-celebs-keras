#!/usr/bin/env python3

import os.path
import glob
from multiprocessing import Pool

import numpy as np
from PIL import Image


NUM_SAMPLES = 202599

proj_root = os.path.split(os.path.dirname(__file__))[0]
images_path = os.path.join(proj_root, 'img_align_celeba_png', '*.png')


def _load_image(f):
    im = Image.open(f) \
              .crop((0, 20, 178, 198)) \
              .resize((64, 64), Image.LANCZOS)
    return np.asarray(im)


def image_loader(batch_size, normalize=True, seed=0, workers=8):
    rng = np.random.RandomState(seed)
    images = glob.glob(images_path)

    while True:
        rng.shuffle(images)
        for s in range(0, len(images), batch_size):
            e = s + batch_size
            batch_names = images[s:e]
            with Pool(workers) as p:
                batch_images = p.map(_load_image, batch_names)
            batch_images = np.stack(batch_images)

            if normalize:
                batch_images = batch_images / 127.5 - 1.
                # To be sure
                batch_images = np.clip(batch_images, -1., 1.)

            # Yield the same batch 3 times since the images will be consumed
            # by three different child generators
            for i in range(3):
                yield batch_images


def discriminator_loader(vae, decoder, img_loader, latent_dim=128, seed=0):
    rng = np.random.RandomState(seed)
    return_real = True
    while True:
        x = next(img_loader)
        batch_size = x.shape[0]

        if return_real:
            inputs = x
            y = np.ones([batch_size, 1], dtype='float32')
        else:
            half_batch = batch_size // 2
            x_tilde = vae.predict(x[half_batch:])
            z_p = rng.normal(size=(half_batch, latent_dim))
            x_p = decoder.predict(z_p)
            inputs = np.concatenate([x_tilde, x_p])
            y = np.zeros([batch_size, 1], dtype='float32')

        # Toggle
        return_real ^= True

        yield inputs, y


def decoder_loader(img_loader, latent_dim=128, seed=0):
    rng = np.random.RandomState(seed)
    while True:
        x = next(img_loader)
        batch_size = x.shape[0]
        z_p = rng.normal(size=(batch_size, latent_dim))
        y_real = np.ones([batch_size, 1], dtype='float32')
        yield [x, z_p], [y_real, y_real]


def encoder_loader(img_loader):
    while True:
        x = next(img_loader)
        yield x, None
