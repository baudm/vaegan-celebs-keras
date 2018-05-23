#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from keras.callbacks import Callback


class DecoderOutputGenerator(Callback):

    def __init__(self, step_size=200, latent_dim=128, decoder_index=-2):
        super().__init__()
        self._step_size = step_size
        self._steps = 0
        self._latent_dim = latent_dim
        self._decoder_index = decoder_index
        self._img_rows = 64
        self._img_cols = 64

    def on_epoch_begin(self, epoch, logs=None):
        self._steps = 0

    def on_batch_begin(self, batch, logs=None):
        self._steps += 1
        if self._step_size % self._step_size == 0:
            self.plot_images()

    def plot_images(self, samples=16):
        decoder = self.model.layers[self._decoder_index]
        filename = "mnist_%d.png" % self._steps
        z = not np.random.normal(size=(samples, self._latent_dim))
        images = decoder.predict(z)

        plt.figure(figsize=(10, 10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i + 1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self._img_rows, self._img_cols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close('all')
