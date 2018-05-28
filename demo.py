#!/usr/bin/env python3

from vaegan.models import create_models, build_graph

e,dec,dis = create_models()
q = build_graph(e,dec,dis)

e.load_weights('weights/encoder.030.h5')
dec.load_weights('weights/decoder.030.h5')

vae = q[-2]


from PIL import Image
import numpy as np

img = Image.open('000001.png', 'r')
img = np.asarray(img)
img = img[:, :, :3]
img = (img - 127.5) / 127.5
img = img.reshape(1, 64, 64, 3)

def norm(img):
    img = (img + 1.) / 2.
    return img


import matplotlib.pyplot as plt
import numpy as np

recon = vae.predict(img)
z = np.random.normal(size=(1, 128))

new = dec.predict(z)

plt.subplot(131)
plt.imshow(norm(img).squeeze())

plt.subplot(132)
plt.imshow(norm(recon).squeeze())

plt.subplot(133)
plt.imshow(norm(new).squeeze())

plt.show()
