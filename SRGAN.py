import tensorflow as tf
from tensorflow.keras.layers import LeakyReLU, Conv2DTranspose, Conv2D, ReLU, PReLU, \
                                    Add, Concatenate, Activation, BatchNormalization, Dense, Flatten
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.models import load_model
from skimage.measure import compare_ssim as structural_similarity
from skimage.measure import compare_psnr as peak_signal_noise_ratio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import cv2
from time import time


def discriminator(image_shape):
    '''
    Build a CNN based discriminator model with input of size image_shape
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
    d_out = Dense(1, activation = 'sigmoid')(d)
    
    disc = Model(input_img, d_out)

    return disc


def resnet_block(n_filters, input_layer):
    '''
    Build a resnet block with n_filters per convolutional layer and attaches it to input_layer
    '''
    # k3n64s1
    g = Conv2D(n_filters, (3,3), strides=(1,1), padding='same')(input_layer)
    g = BatchNormalization()(g)
    g = PReLU()(g)
    # k3n64s1
    g = Conv2D(n_filters, (3,3), strides=(1,1), padding='same')(g)
    g = BatchNormalization()(g)
    
    g = Add()([g, input_layer])
    
    return g


def generator(image_shape, n_resnets):
    '''
    Builds a CNN based generator network with n_resnets resnet blocks and input of size image_shape
    '''
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
    
    return generator

def feature_extractor(i, j):
    '''
    Feature extractor network used to perform content loss
    '''
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    features_list = [layer.output for layer in vgg.layers]
    
    for (k, layer) in zip(range(len(vgg.layers)), vgg.layers):
        if layer.name == "block"+str(i)+"_conv"+str(j):
            break
    model = Model(vgg.input, features_list[k])

    del vgg

    return model
    

entropy_loss = tf.keras.losses.BinaryCrossentropy(from_logits = False)
mse_loss = tf.keras.losses.MeanSquaredError()

def discriminator_loss(real_output, fake_output):
    real_loss = entropy_loss(tf.ones_like(real_output), real_output)
    fake_loss = entropy_loss(tf.zeros_like(fake_output), fake_output)

    return real_loss, fake_loss

def generator_loss(fake_output):
    loss = entropy_loss(tf.ones_like(fake_output), fake_output)
    return loss
        
def content_loss(hr_images, sr_images):
    hr_features = feat_ext(hr_images)
    sr_features = feat_ext(sr_images)
    
    content_loss = mse_loss(hr_features, sr_features)

    return content_loss / 12.75


def progress_update(model, input_img, epoch, factor):
    '''
    Show downsampled image, superresolution image, and ground truth HR image
    '''
    lr_input_img = downSampleAll([input_img], factor)
    prediction = model(lr_input_img, training = False)

    display_list = [lr_input_img[0], prediction[0], input_img]
    titles = ['Downsampled Image', 'Upsampled Image', 'Original Image']

    plt.figure(figsize=(12,12))

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(titles[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')

    plt.tight_layout()
    
    plt.savefig('image_at_epoch{:04d}.png'.format(epoch+1))
    
    print('Saved sample images.')
    plt.close()
    
def show_loss_history(losses, names, title):
    '''
    Show loss history during adversarial training
    '''
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


def train_step_adversarial(lr_images, hr_images):
    '''
    Perform one step of training (i.e. train on a single batch)
    '''
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        sr_images = gen(lr_images, training = True)

        real_output = disc(hr_images, training = True)
        fake_output = disc(sr_images, training = True)
        
        disc_loss_real, disc_loss_fake = discriminator_loss(real_output, fake_output)
        disc_loss = disc_loss_real + disc_loss_fake

        gen_adv_loss = generator_loss(fake_output)
        gen_content_loss = content_loss(hr_images, sr_images)
        gen_total_loss = 0.001 * gen_adv_loss + gen_content_loss

    # Calculate gradients
    gen_gradients = gen_tape.gradient(gen_total_loss, gen.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, disc.trainable_variables)
    
    # Apply gradients
    gen_optimizer.apply_gradients(zip(gen_gradients, gen.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, disc.trainable_variables))
    
    return gen_adv_loss, gen_content_loss, disc_loss_real, disc_loss_fake


def downSample(original_image, factor):
    orig_dim = (original_image.shape[1], original_image.shape[0]) # (width, height)

    width = int(original_image.shape[1] / factor)
    height = int(original_image.shape[0] / factor)
    dim = (width, height)
    downsampled_img = cv2.resize(original_image, dim)

    return downsampled_img

def downSampleAll(images, factor):
    '''
    Downsample images by given factor
    '''
    downsampled = []
    for i in range(len(images)):
        downsampled.append(downSample(images[i], factor))

    return np.asarray(downsampled)


def train_adversarial(train_data, val_data, batch_size, start_epoch, n_epochs, factor):
    '''
    Train generator and discriminator for n_epochs with batch_size
    '''
    # Set aside sample image to track progress
    sample_image = random.choice(val_data)

    # create models directory if doesn't exist (to save models here)
    dirName = 'models'
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    else:    
        print("Directory " , dirName ,  " already exists")
    
    # create valdiation history dataframe
    validation_hist = pd.DataFrame(columns = ['epoch', 'SSIM', 'PSNR'])
    
    # Train for n_epochs
    for epoch in range(start_epoch, n_epochs):
        losses_per_epoch = np.zeros(4)
        start = time()
        print('Starting epoch {}'.format(epoch+1))
        
        n = 0
        for hr_images in train_data:
            lr_images = downSampleAll(hr_images.numpy(), factor)
            losses_per_epoch += train_step_adversarial(lr_images, hr_images)
            n += 1
        

        print('Epoch {} time: {:.2f}'.format(epoch+1, time()-start))
        losses_per_epoch /= n
        
        if epoch == start_epoch:
            losses = np.asarray([losses_per_epoch])
        else:
            losses = np.vstack([losses, losses_per_epoch])

        # update loss history plot
        show_loss_history(losses, ["gen_adv","gen_content", "disc_real", "disc_fake"], title = 'Training History')    
        
        # Show progress update with sample image every 5 epochs
        if (epoch + 1) % 5 == 0:
            progress_update(gen, sample_image, epoch, factor)
        
        # Save models every 10 epochs and compute validation scores
        if (epoch + 1) % 10 == 0:
            # validation metrics
            PSNR, SSIM = 0, 0
            for hr_test_img in val_data:
                lr_test_img = downSampleAll([hr_test_img], factor)
                sr_img = gen(lr_test_img, training = False)

                sr_img = np.asarray(sr_img)

                SSIM += structural_similarity(hr_test_img, sr_img[0], multichannel=True)
                PSNR += peak_signal_noise_ratio(hr_test_img, sr_img[0])
            SSIM /= len(val_data)
            PSNR /= len(val_data)
            
            #validation_hist = validation_hist.append({'epoch':epoch+1, 'SSIM':SSIM, 'PSNR':PSNR}, ignore_index=True)
            validation_hist = validation_hist.append({'epoch':epoch+1, 'SSIM':SSIM, 'PSNR':PSNR}, ignore_index=True)
            
            # Save models
            dirName = 'models/models_at_epoch{:04d}'.format(epoch+1)
            if not os.path.exists(dirName):
                os.mkdir(dirName)
            # save generator
            dirName = 'models/models_at_epoch{:04d}/gen'.format(epoch+1)
            if not os.path.exists(dirName):
                os.mkdir(dirName)
            # Save JSON config to disk
            json_config = gen.to_json()
            with open(dirName + '/architecture.json', 'w') as json_file:
                json_file.write(json_config)
            # Save weights to disk
            gen.save_weights(dirName + '/weights.h5')
            print('Saved generator.')

            # save descriminator
            dirName = 'models/models_at_epoch{:04d}/disc'.format(epoch+1)
            if not os.path.exists(dirName):
                os.mkdir(dirName)
            # Save JSON config to disk
            json_config = disc.to_json()
            with open(dirName + '/architecture.json', 'w') as json_file:
                json_file.write(json_config)
            # Save weights to disk
            disc.save_weights(dirName + '/weights.h5')
            print('Saved discriminator.')
    
    # Write validation history to csv
    validation_hist.to_csv('adv_validation_history.csv')


def loadImages(directory, img_shape, limit = 1200):
    # i = 0
    images = []
    for filename in os.listdir(directory)[0:limit]:
        if filename.endswith('.jpg'): # and i < 10 for testing
            image = load_img(os.path.join(directory,filename), target_size = img_shape)
            image = img_to_array(image)
            images.append(image)

    images = (np.asarray(images) / 127.5) - 1 # scale to [-1,1] range

    return images

############# Dataset preparation #############

factor = 4 # downsampling factor

lr_image_shape = (128,128,3) # low resolution image shape
hr_image_shape = (lr_image_shape[0]*factor, lr_image_shape[1]*factor, 3) # high resolution image shape

train_sz, val_sz, test_sz = 1000, 200, 200 # training, validation and test set sizes

hr_data = loadImages('celeba', img_shape = hr_image_shape, limit = train_sz+val_sz+test_sz)

hr_train_val_data, hr_test_data = train_test_split(hr_data, test_size = test_sz, train_size = train_sz+val_sz, random_state = 10)

hr_train_data, hr_val_data = train_test_split(hr_train_val_data, test_size = val_sz, train_size = train_sz, random_state = 10)

# Make low res datasets for training, validation and testing
lr_train_data = downSampleAll(hr_train_data, factor)
lr_val_data = downSampleAll(hr_val_data, factor)
lr_test_data = downSampleAll(hr_test_data, factor)

del hr_data
del  hr_train_val_data

############# SR-ResNet Training #############

print('Building models')
gen = generator(lr_image_shape, 16) # change number of resnet layers here
gen_optimizer = Adam(lr = 0.0001, beta_1=0.9)

def SSIM_score(y_true, y_pred):
    return tf.image.ssim(y_true*0.5+0.5, y_pred*0.5+0.5, max_val=1.0)

def PSNR_score(y_true, y_pred):
    return tf.image.psnr(y_true*0.5+0.5, y_pred*0.5+0.5, max_val=1.0)

gen.compile(loss = 'mean_squared_error', optimizer = gen_optimizer, metrics = [SSIM_score, PSNR_score])

pretrain_batch_size = 1 # SR-ResNet training batch size
pretrain_epochs = 100 # number of SR-ResNet training epochs
pretrain_history = gen.fit(lr_train_data, hr_train_data, batch_size = pretrain_batch_size, epochs = pretrain_epochs, verbose = 2, validation_data = (lr_val_data, hr_val_data))

# Make plots of training history and save them
idx = range(1, pretrain_epochs+1)
sns.set()
# Plot training and validation loss and metrics
plt.figure(figsize=(15,8))
sns.lineplot(x = idx, y = pretrain_history.history['loss'], label = 'Training loss')
sns.lineplot(x = idx, y = pretrain_history.history['val_loss'], label = 'Validation loss')
plt.title('Model Loss History')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc = 'best')
plt.savefig('loss_history.png')
plt.close()

# Plot training and validation SSIM history
plt.figure(figsize=(15,8))
sns.lineplot(x = idx, y = pretrain_history.history['SSIM_score'], label = 'Training SSIM')
sns.lineplot(x = idx, y = pretrain_history.history['val_SSIM_score'], label = 'Validation SSIM')
plt.title('Model SSIM History')
plt.ylabel('SSIM Score')
plt.xlabel('Epoch')
plt.legend(loc = 'best')
plt.savefig('SSIM_history.png')
plt.close()

# Plot training and validation PSNR history
plt.figure(figsize=(15,8))
sns.lineplot(x = idx, y = pretrain_history.history['PSNR_score'], label = 'Training PSNR')
sns.lineplot(x = idx, y = pretrain_history.history['val_PSNR_score'], label = 'Validation PSNR')
plt.title('Model PSNR History')
plt.ylabel('PSNR Score')
plt.xlabel('Epoch')
plt.legend(loc = 'best')
plt.savefig('PSNR_history.png')
plt.close()

# Save 10 sample images
if not os.path.exists('pretrain_test_samples'):
    os.mkdir('pretrain_test_samples')

for i in range(10):
    hr_img = random.choice(hr_test_data)
    lr_img = downSampleAll([hr_img], factor)
    sr_img = gen(lr_img, training = False)

    dir_name = 'test_samples/sample_{:03d}'.format(i)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    plt.imsave(fname = dir_name + '/lr_img.png', arr = lr_img[0] * 0.5 + 0.5)
    plt.imsave(fname = dir_name + '/sr_img.png', arr = sr_img[0] * 0.5 + 0.5)
    plt.imsave(fname = dir_name + '/hr_img.png', arr = hr_img * 0.5 + 0.5)

# Save model
dirName = 'models'
if not os.path.exists(dirName):
    os.mkdir(dirName)
dirName = 'models/gen_initial'
if not os.path.exists(dirName):
    os.mkdir(dirName)

# Save JSON config to disk
json_config = gen.to_json()
with open(dirName + '/architecture.json', 'w') as json_file:
    json_file.write(json_config)
# Save weights to disk
gen.save_weights(dirName + '/weights.h5')
print('Saved pre-trained generator.')

############# SRGAN training for fine tuning #############

bufferSize = 1000
batch_size = 1
print('Converting to tf dataset')
hr_train_data = tf.data.Dataset.from_tensor_slices(np.asarray(hr_train_data)).shuffle(bufferSize).batch(batch_size)

disc = discriminator((lr_image_shape[0]*factor, lr_image_shape[1]*factor, 3))
disc_optimizer = Adam(lr = 0.0001, beta_1=0.9)

print('Started feature extractor.')
i = 5
j = 4
feat_ext = feature_extractor(i, j)

start_epoch = 0
n_epochs_adv = 50 # input number of adversarial training epochs

train_adversarial(hr_train_data, hr_val_data, batch_size, start_epoch, n_epochs_adv, factor)

print("SRGAN TRAINING COMPLETE!")

############# Testing #############

validation_hist = pd.read_csv('adv_validation_history.csv')

def normalize_column(df, column_name):
    data = df[column_name].to_numpy().reshape(-1, 1)
    print(type(data))
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    df[column_name] = scaled_data
    
    return df

validation_hist = normalize_column(validation_hist, 'SSIM')
validation_hist = normalize_column(validation_hist, 'PSNR')

best_epoch_SSIM = validation_hist[validation_hist.SSIM == validation_hist.SSIM.max()]
best_epoch_PSNR = validation_hist[validation_hist.PSNR == validation_hist.PSNR.max()]

if best_epoch_PSNR.epoch.iloc[0] == best_epoch_SSIM.epoch.iloc[0]:
    best_epoch  = best_epoch_SSIM.epoch.iloc[0]
else:
    combined_SSIM_epoch = best_epoch_SSIM['SSIM'] + best_epoch_SSIM['PSNR']
    combined_PSNR_epoch = best_epoch_PSNR['SSIM'] + best_epoch_PSNR['PSNR']
    if combined_SSIM_epoch.iloc[0] > combined_PSNR_epoch.iloc[0]:
        best_epoch = best_epoch_SSIM.epoch.iloc[0]
    else:
        best_epoch = best_epoch_PSNR.epoch.iloc[0]
best_epoch = int(best_epoch)

# Load in best model
model_path = 'models/models_at_epoch{:04d}'.format(best_epoch) + '/gen'
with open(model_path + '/architecture.json') as json_file:
    json_config = json_file.read()
best_model = tf.keras.models.model_from_json(json_config)
best_model.load_weights(model_path + '/weights.h5')
print('Successfully loaded best model.')

# Compute SSIM and PSNR on test dataset
PSNR, SSIM = 0, 0
for lr_img, hr_img in zip(lr_test_data, hr_test_data):
    lr_img = np.expand_dims(lr_img, axis=0)
    sr_img = best_model(lr_img, training = False)
    sr_img = np.asarray(sr_img)

    SSIM += structural_similarity(hr_img, sr_img[0], multichannel=True)
    PSNR += peak_signal_noise_ratio(hr_img, sr_img[0])
SSIM /= len(hr_test_data)
PSNR /= len(hr_test_data)

print('Test set SSIM: {:.2f}'.format(SSIM))
print('Test set PSNR: {:.2f}'.format(PSNR))

# Save 10 sample images
if not os.path.exists('test_samples'):
    os.mkdir('test_samples')

for i in range(10):
    hr_img = random.choice(hr_test_data)
    lr_img = downSampleAll([hr_img], factor)
    sr_img = best_model(lr_img, training = False)

    dir_name = 'test_samples/sample_{:03d}'.format(i)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    plt.imsave(fname = dir_name + '/lr_img.png', arr = lr_img[0] * 0.5 + 0.5)
    plt.imsave(fname = dir_name + '/sr_img.png', arr = sr_img[0] * 0.5 + 0.5)
    plt.imsave(fname = dir_name + '/hr_img.png', arr = hr_img * 0.5 + 0.5)

print('DONE EVERYTHING!!!!!!!!!')

