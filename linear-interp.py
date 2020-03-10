import cv2
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
import numpy as np
import itertools
from scipy.interpolate import griddata

image = Image.open("image.jpeg")
image = asarray(image)

pyplot.imshow(image)
width = image.shape[0]
height = image.shape[1]

[r,g,b] = cv2.split(image)

res = 4

points = np.array(list(itertools.product(np.arange(0,width,1),np.arange(0,height,1))))
interpoints = list(itertools.product(np.arange(0,width,1/res),np.arange(0,height,1/res)))

gridr = griddata(points, r.reshape((width*height)), interpoints, method='cubic')
gridg = griddata(points, g.reshape((width*height)), interpoints, method='cubic')
gridb = griddata(points, b.reshape((width*height)), interpoints, method='cubic')

gridr = gridr.reshape((width*res, height*res))
gridg = gridg.reshape((width*res, height*res))
gridb = gridb.reshape((width*res, height*res))

imageUpsamp = cv2.merge((gridr, gridg, gridb))

pyplot.imshow(imageUpsamp/255)

img = Image.fromarray(image)
img.save("sample_data/upscaled.jpeg")