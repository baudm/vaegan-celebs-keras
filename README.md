# vaegan-celebs-keras
EE 298 Group 6:


Darwin Bautista | Paolo Valdez

Hello!

This is our implementation of "Autoencoding beyond pixels using a learned similarity metric" by Larsen et al. (https://arxiv.org/abs/1512.09300)
Many of the implementation details (which were unclear in the paper) were directly lifted from the authors' [official code](https://github.com/andersbll/autoencoding_beyond_pixels)

# Prequisites

Tensorflow (>=1.4)

Keras (>= 2.1.4)

OpenCV (>= 3.4.0)

Numpy


# Dataset
Available @ http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

# Presenstation Slides
Our Presentation Material can be found at: 
https://docs.google.com/presentation/d/1PhjrLkPf-UstSI_oZXod0qb4k_HsPbtNlm9ekJE4gvY/edit?usp=sharing

# Pre-trained weights:
https://drive.google.com/open?id=1ELiB3GNeT_I9RyTqpeoNNQ2CoOZ-v81X

# Data sample:

![alt text](https://github.com/baudm/vaegan-celebs-keras/blob/master/celeba_dataset.jpg "Logo Title Text 1")
 
# Variational Auto Encoder (VAE) Architecture
Encoder Training Model

![alt text](https://github.com/baudm/vaegan-celebs-keras/blob/master/training_model_encoder.jpg "Logo Title Text 1")

Decoder Training Model
![alt text](https://github.com/baudm/vaegan-celebs-keras/blob/master/training_model_decoder.jpg "Logo Title Text 1")

# General Adverserial Network (GAN) Architecture

Discriminator Training Model
![alt text](https://github.com/baudm/vaegan-celebs-keras/blob/master/training_model_discriminator.jpg " GAN ")

# Generated Image Results:

![Final Results](final-results.png)
