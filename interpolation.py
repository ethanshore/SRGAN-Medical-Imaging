import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage.measure import compare_ssim as structural_similarity
from skimage.measure import compare_psnr as peak_signal_noise_ratio
import os

def interpolate_image(path, print_pics = False):
	orig_img = plt.imread(path)
	print('Original Dimensions : ', orig_img.shape)
	orig_dim = (orig_img.shape[1], orig_img.shape[0]) # (width, height)

	scale_percent = 20 # percent of original size
	width = int(orig_img.shape[1] * scale_percent / 100)
	height = int(orig_img.shape[0] * scale_percent / 100)
	dim = (width, height)
	downsampled_img = cv2.resize(orig_img, dim, interpolation = cv2.INTER_AREA)

	print('Downsampled Dimensions : ', downsampled_img.shape)

	# Upscale image
	resized_NN = cv2.resize(downsampled_img, orig_dim, interpolation = cv2.INTER_NEAREST)
	resized_BL = cv2.resize(downsampled_img, orig_dim, interpolation = cv2.INTER_LINEAR)
	resized_BC = cv2.resize(downsampled_img, orig_dim, interpolation = cv2.INTER_CUBIC)

	print('Resized Dimensions : ', resized_NN.shape)
	if print_pics:
		plt.figure(figsize=(16,10))

		plt.subplot(2,3,1)
		plt.imshow(orig_img)
		# plt.imsave('orig.jpg', orig_img)
		plt.title('Original Image')
		plt.axis('off')

		plt.subplot(2,3,2)
		plt.imshow(downsampled_img)
		plt.title('Downsampled Image')
		plt.axis('off')

		plt.subplot(2,3,4)
		plt.imshow(resized_NN)
		# plt.imsave('NN.jpg', resized_NN)
		plt.title('Upsampled Image - Nearest Neighbour')
		plt.axis('off')

		plt.subplot(2,3,5)
		plt.imshow(resized_BL)
		# plt.imsave('BL.jpg', resized_BL)
		plt.title('Upsampled Image - Bilinear')
		plt.axis('off')

		plt.subplot(2,3,6)
		plt.imshow(resized_BC)
		# plt.imsave('BC.jpg', resized_BC)
		plt.title('Upsampled Image - Bicubic')
		plt.axis('off')

		plt.tight_layout()
		plt.savefig('interpolated_' + path[:-4] + '.png')
		plt.show()

	NN_ssim = structural_similarity(orig_img, resized_NN, multichannel=True)
	NN_psnr = peak_signal_noise_ratio(orig_img, resized_NN)

	BL_ssim = structural_similarity(orig_img, resized_BL, multichannel=True)
	BL_psnr = peak_signal_noise_ratio(orig_img, resized_BL)

	BC_ssim = structural_similarity(orig_img, resized_BC, multichannel=True)
	BC_psnr = peak_signal_noise_ratio(orig_img, resized_BC)

	return NN_ssim, NN_psnr, BL_ssim, BL_psnr, BC_ssim, BC_psnr

NN_ssim, NN_psnr = list(), list()
BL_ssim, BL_psnr = list(), list()
BC_ssim, BC_psnr = list(), list()

directory = 'medical_images' # change path to images here
for filename in os.listdir(directory):
	if filename.endswith('.jpg'):
		path = os.path.join(directory, filename)
		a, b, c, d, e, f = interpolate_image(path, print_pics=True)
		NN_ssim.append(a)
		NN_psnr.append(b)
		BL_ssim.append(c)
		BL_psnr.append(d)
		BC_ssim.append(e)
		BC_psnr.append(f)
	else:
		continue

print('Nearest Neighbour Interpolation:')
print('\tSSIM: {:.2f}'.format(np.mean(NN_ssim)))
print('\tPSNR: {:.2f}'.format(np.mean(NN_psnr)))

print('Bilinear Interpolation:')
print('\tSSIM: {:.2f}'.format(np.mean(BL_ssim)))
print('\tPSNR: {:.2f}'.format(np.mean(BL_psnr)))

print('Bicubic Interpolation:')
print('\tSSIM: {:.2f}'.format(np.mean(BC_ssim)))
print('\tPSNR: {:.2f}'.format(np.mean(BC_psnr)))