#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, Conv2DTranspose, Conv2D, ReLU, PReLU, Add, Concatenate, Activation, BatchNormalization, Dense, Flatten
from tensorflow.keras import Model, Input
import numpy as np
import matplotlib.pyplot as plt
import os
from time import time
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import load_img, img_to_array

# from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from skimage.measure import compare_ssim as structural_similarity
from skimage.measure import compare_psnr as peak_signal_noise_ratio
from sklearn.model_selection import train_test_split


import pandas as pd
import random
import seaborn as sns
from PIL import Image
import cv2
import sys


# In[55]:


def discriminator(image_shape):
    '''
    Build a 70x70 PatchGAN discriminator
    '''
    input_img = Input(shape = image_shape)
    # k3n64s1
    d = Conv2D(64, (3,3), strides = (1,1), padding = 'same' )(input_img)
    d = LeakyReLU(0.2)(d)
    
    # k3n64s2
    d = Conv2D(64, (3,3), strides = (2,2), padding = 'same' )(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(0.2)(d)
    
    # k3n128s1
    d = Conv2D(128, (3,3), strides = (1,1), padding = 'same' )(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(0.2)(d)
    
    # k3n128s2
    d = Conv2D(128, (3,3), strides = (2,2), padding = 'same' )(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(0.2)(d)
    
    # k3n256s1
    d = Conv2D(256, (3,3), strides = (1,1), padding = 'same' )(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(0.2)(d)
    
    # k3n256s2
    d = Conv2D(256, (3,3), strides = (2,2), padding = 'same' )(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(0.2)(d)
    
    # k3n512s1
    d = Conv2D(512, (3,3), strides = (1,1), padding = 'same' )(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(0.2)(d)
    
    # k3n512s2
    d = Conv2D(512, (3,3), strides = (2,2), padding = 'same' )(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(0.2)(d)
    
    d = Flatten()(d)
    # d = Dense(1024)(d)
    # d = LeakyReLU(0.2)(d)

    d_out = Dense(1, activation = 'sigmoid')(d)
    
    disc = Model(input_img, d_out)

    return disc

def resnet_block(n_filters, input_layer):
    # k3n64s1
    g = Conv2DTranspose(n_filters, (3,3), strides=(1,1), padding='same')(input_layer)
    g = BatchNormalization()(g)
    g = PReLU()(g)
    # k3n64s1
    g = Conv2DTranspose(n_filters, (3,3), strides=(1,1), padding='same')(g)
    g = BatchNormalization()(g)
    
    # g = Concatenate()([g, input_layer])
    g = Add()([g, input_layer])
    
    return g


def generator(image_shape, n_resnets):

    input_image = Input(shape=image_shape)
    
    g = Conv2D(64, (3,3), strides=(1,1), padding='same')(input_image)
    g = PReLU()(g)
    
    g_res = resnet_block(64, g)
    for i in range(n_resnets - 1):
        g_res = resnet_block(64, g_res)

    g_res = Conv2D(64, (3,3), strides=(1,1), padding='same')(g_res)
    g_res = BatchNormalization()(g_res)
    
    g = Add()([g, g_res])
        
    g = Conv2D(256, (3,3), strides=(1,1), padding='same')(g)
    g = tf.nn.depth_to_space(g, 2)
    g = PReLU()(g)

    g = Conv2D(256, (3,3), strides=(1,1), padding="same")(g)
    g = tf.nn.depth_to_space(g, 2)
    g = PReLU()(g)

    g = Conv2D(3, (9,9), strides=(1,1), padding="same")(g)
    
    output_image = Activation("tanh")(g)

    generator = Model(input_image, output_image)
    print(generator.output_shape)
    
    return generator

def feature_extractor(i, j):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    features_list = [layer.output for layer in vgg.layers]
    
    for (k, layer) in zip(range(len(vgg.layers)), vgg.layers):
        if layer.name == "block"+str(i)+"_conv"+str(j):
            break
    model = Model(vgg.input, features_list[k])

    print('Model size:', sys.getsizeof(model))
    print('vgg size:', sys.getsizeof(vgg))
    del vgg

    return model
    

entropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits = False)
mse_loss = tf.keras.losses.MeanSquaredError()

def discriminator_loss(real_output, fake_output):
    real_loss = entropy_loss(tf.ones_like(real_output), real_output)
    fake_loss = entropy_loss(tf.zeros_like(fake_output), fake_output)

    total_loss = real_loss + fake_loss

    return total_loss

def generator_loss(fake_output):
    loss = entropy_loss(tf.ones_like(fake_output), fake_output)
    return loss
        
def content_loss(hr_images, sr_images):
    
    hr_features = feat_ext(hr_images)
    sr_features = feat_ext(sr_images)
    
    content_loss = mse_loss(hr_features, sr_features)

    return content_loss / 12.75


# In[56]:


def progress_update(model, input_img, epoch):
    lr_input_img = downSampleAll([input_img], 4)
    prediction = model(lr_input_img, training = False)

    display_list = [lr_input_img[0], prediction[0], input_img]
    titles = ['Downsampled Image', 'Upsampled Image', 'Original Image']

    plt.figure(figsize=(12,12))

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(titles[i])
        plt.imshow(display_list[i] * 0.5 + 0.5) # scale images back from [-1,1] range to [0,1] for plotting
        plt.axis('off')

    plt.tight_layout()
    
    plt.savefig('image_at_epoch{:04d}.png'.format(epoch+1))
    
    print('Saved sample images.')
    plt.close()
    
def show_loss_history(losses, names, title):
    epoch_idx = range(1, len(losses)+1)
    plt.figure(figsize=(15,12))
    sns.set()
    for i in range(len(names)):
        sns.lineplot(x = epoch_idx, y = losses[:,i], label = names[i])
    plt.xlabel('Epoch')
    plt.legend(loc = 'best')
    plt.title(title)

    plt.savefig(title + '.png')
    print('Saved ' + title)

    plt.close()


# In[61]:


def train_step(lr_images, hr_images):
#     with tf.GradientTape() as gen_adv_tape, tf.GradientTape() as gen_content_tape, tf.GradientTape() as disc_tape:
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        sr_images = gen(lr_images, training = True)

        real_output = disc(hr_images, training = True)
        fake_output = disc(sr_images, training = True)
        
        disc_loss = discriminator_loss(real_output, fake_output)
        
        gen_adv_loss = generator_loss(fake_output)
        gen_content_loss = content_loss(hr_images, sr_images)
        gen_total_loss = gen_adv_loss + gen_content_loss

    # Calculate gradients
    # gen_adv_gradients = gen_adv_tape.gradient(gen_adv_loss, gen.trainable_variables)
    # gen_content_gradients = gen_content_tape.gradient(gen_content_loss, gen.trainable_variables)
    
    gen_gradients = gen_tape.gradient(gen_total_loss, gen.trainable_variables)
    
    disc_gradients = disc_tape.gradient(disc_loss, disc.trainable_variables)
    
    # Apply gradients
    # gen_optimizer.apply_gradients(zip(gen_adv_gradients, gen.trainable_variables))
    # gen_optimizer.apply_gradients(zip(gen_content_gradients, gen.trainable_variables))
    
    gen_optimizer.apply_gradients(zip(gen_gradients, gen.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, disc.trainable_variables))
    
    return gen_adv_loss, gen_content_loss, disc_loss


# In[77]:


def downSample(original_image, factor):
    orig_dim = (original_image.shape[1], original_image.shape[0]) # (width, height)

    width = int(original_image.shape[1] / factor)
    height = int(original_image.shape[0] / factor)
    dim = (width, height)
    downsampled_img = cv2.resize(original_image, dim) #, interpolation = cv2.INTER_AREA)

    # print('DwnSmplImg size:', sys.getsizeof(downsampled_img))

    return downsampled_img


def downSampleAll(images, factor):

    # downsampled = [images[0]] * len(images)
    downsampled = []

    for i in range(len(images)):
        downsampled.append(downSample(images[i], factor))
        # downsampled[i] = downSample(images[i], factor)

    # print(np.asarray(downsampled).shape)

    return np.asarray(downsampled)


# In[78]:


def train(train_data, test_data, batch_size, start_epoch, n_epochs, factor):
    sample_image = random.choice(test_data)
    # create models directory if doesn't exist
    dirName = 'models'
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    else:    
        print("Directory " , dirName ,  " already exists")
    
    validation_hist = pd.DataFrame(columns = ['epoch', 'SSIM', 'PSNR'])
    
    for epoch in range(start_epoch, n_epochs):
        losses_per_epoch = np.zeros(3)
        start = time()
        print('Starting epoch {}'.format(epoch+1))
        
        n = 0
        for hr_images in train_data:
            lr_images = downSampleAll(hr_images.numpy(), factor)
            losses_per_epoch += train_step(lr_images, hr_images)
            n += 1
        

        print('Epoch {} time: {:.2f}'.format(epoch+1, time()-start))
        losses_per_epoch /= n
        
        if epoch == start_epoch:
            losses = np.asarray([losses_per_epoch])
        else:
            losses = np.vstack([losses, losses_per_epoch])

        show_loss_history(losses, ["gen_adv","gen_content", "disc"], title = 'Training History')    
            
        if (epoch + 1) % 5 == 0:
            progress_update(gen, sample_image, epoch)
        
        # save models every 10 epochs
        if (epoch + 1) % 10 == 0:
            # validation metrics
            PSNR, SSIM = 0, 0
            for hr_test_img in test_data:
                lr_test_img = downSampleAll([hr_test_img], factor)
                sr_img = gen(lr_test_img, training = False)

                sr_img = np.asarray(sr_img)

                SSIM += structural_similarity(hr_test_img, sr_img[0], multichannel=True)
                PSNR += peak_signal_noise_ratio(hr_test_img, sr_img[0])
            SSIM /= len(test_data)
            PSNR /= len(test_data)
            
            #validation_hist = validation_hist.append({'epoch':epoch+1, 'SSIM':SSIM, 'PSNR':PSNR}, ignore_index=True)
            validation_hist = validation_hist.append({'epoch':epoch+1, 'SSIM':SSIM, 'PSNR':PSNR}, ignore_index=True)
            
            # create directory for epoch
            dirName = 'models/models_at_epoch{:04d}'.format(epoch+1)
            if not os.path.exists(dirName):
                os.mkdir(dirName)
                
            gen.save(dirName + '/gen')
            disc.save(dirName + '/disc')
    
    #print(validation_hist)
    validation_hist.to_csv('validation_history.csv')


# In[1]:


def loadImages(directory, img_shape, limit = 1200):
    # i = 0
    images = []
    for filename in os.listdir(directory)[0:limit]:
        if filename.endswith('.jpg'): # and i < 10 for testing
            image = load_img(os.path.join(directory,filename), target_size = img_shape)
            image = img_to_array(image)
            # image = plt.imread(os.path.join(directory,filename))
            # image = cv2.resize(image, (img_shape[0], img_shape[1]))
            # image = (image - 127.5)/127.5 
            images.append(image)

    images = (np.asarray(images) / 127.5) - 1

    return images


# In[75]:
factor = 4

lr_image_shape = (128,128,3)
hr_image_shape = (lr_image_shape[0]*factor, lr_image_shape[1]*factor, 3)

hr_data = loadImages('celeba', img_shape = hr_image_shape, limit = 1200)

sample_image = random.choice(hr_data)
plt.imsave('sample_img.png', sample_image * 0.5 + 0.5)

downsampled_img_sample = downSampleAll([sample_image], factor)[0]  * 0.5 + 0.5
plt.imsave('downsampled_sample_img.png', downsampled_img_sample)

hr_train_data, hr_test_data = train_test_split(hr_data, test_size = 200, train_size = 1000, random_state = 10)

del hr_data

bufferSize = 1000
batch_size = 1
print('Converting to tf dataset')
hr_train_data = tf.data.Dataset.from_tensor_slices(np.asarray(hr_train_data)).shuffle(bufferSize).batch(batch_size)

print('Building models')
gen = generator(lr_image_shape, 16)
disc = discriminator((lr_image_shape[0]*factor, lr_image_shape[1]*factor, 3))

gen_optimizer = Adam(lr = 0.0001, beta_1=0.9)
disc_optimizer = Adam(lr = 0.0001, beta_1=0.9)

print('Started feature extractor.')
i = 5
j = 4
feat_ext = feature_extractor(i, j) 

start_epoch = 0
n_epochs = 100 #100

train(hr_train_data, hr_test_data, batch_size, start_epoch, n_epochs, factor)

print("SRGAN COMPLETE!")

# In[ ]:




