import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Input, Model
from keras.layers import Dense, Add, LeakyReLU, PReLU, Activation, BatchNormalization, Dense, Flatten
from keras.layers import Conv2D, UpSampling2D
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, load_img
from keras.applications import vgg19
from keras.callbacks import TensorBoard
from scipy.misc import imsave
import glob
import time
import os

# Residual block:
def res_block(g_layer):
	# k3n64s1
	r = Conv2D(64, (3,3), 1, padding='same')(g_layer)
	r = BatchNormalization()(r)
	r = PReLU()(r)
	# k3n64s1
	r = Conv2D(64, (3,3), 1, padding='same')(r)
	r = BatchNormalization()(r)
	# Elementwise sum
	r = Add()([g_layer, r])

	return r

def build_generator(shape_input = (64,64,3), n_resnet = 16):
	# Hyperparameter definition:
	n_resnet = 16  # should we have less?
	momentum = 0.8 

	input_layer = Input(shape=shape_input)

	g = Conv2D(64, (9,9), 1, padding='same')(input_layer)
	g = PReLU()(g)

	g_res = res_block(g)
	for _ in range(n_resnet-1):
		g_res = res_block(g_res)

	# k3n64s1
	g_res = Conv2D(64, (3,3), 1, padding='same')(g_res)
	g_res = BatchNormalization()(g_res)
	# Elementwise sum
	g = Add()([g, g_res])
	# k3n256s1
	g = Conv2D(256, (3,3), 1, padding='same')
	g = PixelShuffleShit # Pixel shuffle here
	g = PReLU()(g)
	# k3n256s1
	g = Conv2D(256, (3,3), 1, padding='same')
	g = PixelShuffleShit # Pixel shuffle here
	g = PReLU()(g)
	# k9n3s1
	g = 



	generator2 = Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(residual)
    generator2 = BatchNormalization(momentum=momentum)(generator2)

    generator3 = Add()([generator2, generator1])

    generator4 = UpSampling2D(size=2)(generator3)
   	generator4 = Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(generator4)
    generator4 = Activation("relu")(generator4)

    generator5 = UpSampling2D(size=2)(generator4)
    generator5 = Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(generator5)
    generator5 = Activation("relu")(generator5)

    generator6 = Conv2D(filters=3, kernel_size=9, strides=1, padding="same")(generator5)
    output_layer = Activation("tanh")(generator6)

    model = Model(inputs=[input_layer], outputs=[output_layer], name="generator")

    return model

def build_discriminator(input_shape = (256, 256, 3)):
	input_img = Input(shape = input_shape)
	# k3n64s1
	d = Conv2D(64, (3,3), 1, padding='same')(input_img)
	d = LeakyReLU(0.2)(d)
	# k3n64s2
	d = Conv2D(64, (3,3), 2, padding='same')(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(0.2)(d)
	# k3n128s1
	d = Conv2D(128, (3,3), 1, padding='same')(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(0.2)(d)
	# k3n128s2
	d = Conv2D(128, (3,3), 2, padding='same')(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(0.2)(d)
	# k3n256s1
	d = Conv2D(256, (3,3), 1, padding='same')(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(0.2)(d)
	# k3n256s2
	d = Conv2D(256, (3,3), 2, padding='same')(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(0.2)(d)
	# k3n512s1
	d = Conv2D(512, (3,3), 1, padding='same')(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(0.2)(d)
	# k3n512s2
	d = Conv2D(512, (3,3), 2, padding='same')(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(0.2)(d)
	# Fully connected layers
	d = Flatten()(d)
	d = Dense(1024)(d)
	d = LeakyReLU(0.2)(d)

	d_out = Dense(1, activation = 'sigmoid')

	model = Model(input_img, d_out, name = 'discriminator')

	return model

def build_combined_model(g_model, d_model, vgg19, input_shape = (256, 256, 3)):
	d_model.trainable = False

	input_img = Input(shape = input_shape)
	g_out = g_model(input_img)

	d_out = d_model(g_out)
	vgg_out = vgg19(g_out)

	model = Model(input_img, [d_out, vgg_out])

	return model

# Training params
image_directory = "image/directory"

n_epochs = 20000
batchSize = 1

lr_input_shape = (64, 64, 3)
hr_input_shape = tuple([4 * x for x in lr_input_shape])
# Define model optimizers
g_opt = Adam(lr = 0.0002, beta_1 = 0.5)
d_opt = Adam(lr = 0.0002, beta_1 = 0.5)
# define models
generator = build_generator(lr_input_shape)
discriminator = build_discriminator(hr_input_shape)

vgg = build_vgg()
vgg.trainable = False
vgg.compile(loss="mse", optimizer=optimizer, metrics=["accuracy"])   # Maybe we should different loss and metrics?

discriminator = build_discriminator()
discriminator.compile(loss="mse", optimizer=optimizer, metrics=["accuracy"])   # Maybe we should different loss and metrics?

generator = build_gen()

lr_input_layer = Input(shape=lr_input_shape)
hr_input_layer = Input(shape=hr_input_shape)

hr_generated_img = generator(lr_input_layer)

features = vgg(high_res_generated_img)

discriminator.trainable = False

probability_is_real = discriminator(hr_generated_img)

adversarial_model = Model([lr_input_layer, hr_input_layer], [probability_is_real, features])
adversarial_model.compile(loss=["binary_crossentropy", "mse"], loss_weights=[1e-3, 1], optimizer=optimizer)
