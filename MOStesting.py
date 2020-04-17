import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import os
from keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import MinMaxScaler

def loadImages(directory, img_shape, limit = 10):
    # i = 0
    images = []
    for filename in random.sample(os.listdir(directory), limit):
        if filename.endswith('.jpg'): # and i < 10 for testing
            image = load_img(os.path.join(directory,filename), target_size = img_shape)
            image = img_to_array(image)
            images.append(image)

    images = np.asarray(images) # note we are loading them in with normal range, not [-1,1]

    return images

def downSample(original_image, factor):
    orig_dim = (original_image.shape[1], original_image.shape[0]) # (width, height)

    width = int(original_image.shape[1] / factor)
    height = int(original_image.shape[0] / factor)
    dim = (width, height)
    downsampled_img = cv2.resize(original_image, dim)

    return downsampled_img

def downSampleAll(images, factor):
    downsampled = []
    for i in range(len(images)):
        downsampled.append(downSample(images[i], factor))

    return np.asarray(downsampled)

# Read in 10 random images and downsample them
hr_img_shape = (512, 512, 3)
lr_img_shape = (hr_img_shape[0]/4, hr_img_shape[1]/4, 3)

path = 'brainScans'
hr_imgs = loadImages(path, hr_img_shape, limit = 10)
lr_imgs = downSampleAll(hr_imgs, 4)

# Load SR-ResNet and SRGAN (at best epoch)
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

# Load SR-ResNet and SRGAN (at best epoch)

print(best_epoch)

model_path = 'models/models_at_epoch00' + str(best_epoch) + "/gen"
with open(model_path + '/architecture.json') as json_file:
    json_config = json_file.read()

srgan_model = tf.keras.models.model_from_json(json_config)
srgan_model.load_weights(model_path + '/weights.h5')
print('Successfully loaded best model.')

model_path = 'models/gen_initial'
with open(model_path + '/architecture.json') as json_file:
    json_config = json_file.read()
    
srres_model = tf.keras.models.model_from_json(json_config)
srres_model.load_weights(model_path + '/weights.h5')
print('Successfully loaded best model.')

if not os.path.exists('test_samples'):
    os.mkdir('test_samples')

# Create generated images and plots
sample_num = 0
n_plots = 3
fig, ax = plt.subplots(n_plots, 5, figsize = (5*4, 4*n_plots))

for lr_img, hr_img in zip(lr_imgs, hr_imgs):
    bc_interp = cv2.resize(lr_img, hr_img_shape, interpolation = cv2.INTER_CUBIC)

    img_for_models = (np.expand_dims(lr_img, axis=0) / 127.5) - 1 # scale to [-1,1] range
    srgan_img = srgan_model(img_for_models, training = False)[0]
    srres_img = srres_model(img_for_models, training = False)[0]

    dir_name = 'test_samples/sample_{:03d}'.format(sample_num)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    plt.imsave(fname = dir_name + '/lr_img.png', arr = lr_img * 0.5 + 0.5)
    plt.imsave(fname = dir_name + '/hr_img.png', arr = hr_img * 0.5 + 0.5)

    plt.imsave(fname = dir_name + '/bicubic.png', arr = bc_interp * 0.5 + 0.5)
    plt.imsave(fname = dir_name + '/srresnet.png', arr = srres_img[0] * 0.5 + 0.5)
    plt.imsave(fname = dir_name + '/srgan.png', arr = srgan_img[0] * 0.5 + 0.5)


    if sample_num < n_plots:
        # Create plot
        display_list = [lr_img, bc_interp, srres_img[0], srgan_img[0], hr_img]
        titles = ['Downsampled Image', 'Bicubic Interpolation', 'SR-ResNet', 'SRGAN', 'Original Image']

        for i in range(5):
            ax[sample_num, i].imshow(display_list[i] * 0.5 + 0.5)
            ax[sample_num, i].axis('off')
            if sample_num == 0:
                ax[sample_num, i].set_title(titles[i])

plt.tight_layout()
    
plt.savefig('final_image_samples.png', dpi=600/(4*n_plots))
print('Saved sample images.')

plt.close()