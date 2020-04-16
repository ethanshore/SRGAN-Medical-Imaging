import numpy as np
from scipy.ndimage import gaussian_filter
from keras.preprocessing.image import load_img, img_to_array
import os
import matplotlib.pyplot as plt
from skimage.color import rgb2gray


def getPixelDiff(image):
    pixel_diff = 0
    
    # get the pixel difference between adjacent pixels within rows of the image array
    for row in image:
        last = row[0]
        for col in row[1:]:
            pixel_diff += abs(last - col)
            last = col.copy()
            
    return pixel_diff.sum()


def filterImages(images):
    pixel_diff_scores = np.zeros(len(images))

    for i, image in enumerate(images):
        # smooth the image using a gaussian filter to remove noise
        smoothed = gaussian_filter(image, sigma=5)
        
        # sum the pixel differences for both the image and its tranpose to get the total
        pixel_diff_scores[i] = getPixelDiff(smoothed) + getPixelDiff(np.transpose(smoothed))

    mean_score = np.mean(pixel_diff_scores)
    images_to_use = []
    
    # only save images that have higher than the mean pixel difference score
    for i, score in enumerate(pixel_diff_scores):
        if(score > mean_score):
            plt.imsave("medical_images/"+str(i)+".jpg", (images[i]+1)/2)


def loadImages(directory, img_shape, limit = 5000):
    images = []
    for filename in os.listdir(directory)[0:limit]:
        if filename.endswith('.jpg'): # and i < 10 for testing
            image = load_img(os.path.join(directory,filename), target_size = img_shape)
            image = img_to_array(image)
            images.append(image)

    images = (np.asarray(images) / 127.5) - 1 # scale to [-1,1] range

    return images


images = loadImages("Dataset", (512,512,3)) # change "Dataset" to the name of the folder containing the images

filterImages(images)

