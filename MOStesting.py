import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
lr_img_shape = (hr_image_shape[0]/4, hr_image_shape[1]/4, 3)

path = 'path_to_images' ##### add model path here #####
hr_imgs = loadImages(path, hr_img_shape, limit = 10)
lr_imgs = downSampleAdd(hr_imgs, 4)

# Load SR-ResNet and SRGAN (at best epoch)

srgan_model = ###### add model here #######
srres_model = ###### add model here #######

# Create generated images and plots
sample_num = 1
for lr_img, hr_img in zip(lr_imgs, hr_imgs):
    bc_interp = cv2.resize(lr_img, hr_img_shape, interpolation = cv2.INTER_CUBIC)

    img_for_models = (np.expand_dims(lr_img, axis=0) / 127.5) - 1 # scale to [-1,1] range
    srgan_img = srgan_model(img_for_models, training = False)[0]
    srres_img = srres_model(img_for_models, training = False)[0]

    sr_imgs = [bc_interp, srres_img, srgan_img]
    disp_titles = ['Bicubic Interpolation', 'SRResNet', 'SRGAN']

    disp_order = random.shuffle([0,1,2])
    display_list = [hr_img] + [sr_imgs[i] for i in disp_order]
    titles_labeled = ['Ground Truth'] + [disp_titles[i] for i in disp_order]
    titles_unlabeled = ['Ground Truth', 'Generated Image A', 'Generated Image B', 'Generated Image C']
    
    # Unlabeled plot
    plt.figure(figsize=(12,12))
    for i in range(4)
        plt.subplot(1, len(display_list), i+1)
        plt.title(titles_unlabeled[i])
        if i == 0 or disp_order[i-1] == 0:
            plt.imshow(display_list[i])
        else:
            plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('unlabeled_test_{:03d}.png'.format(sample_num))
    plt.close()

    # Labeled plot
    plt.figure(figsize=(12,12))
    for i in range(4)
        plt.subplot(1, len(display_list), i+1)
        plt.title(titles_labeled[i])
        if i == 1 or disp_order[i] == 0:
            plt.imshow(display_list[i])
        else:
            plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('labeled_test_{:03d}.png'.format(sample_num))
    plt.close()