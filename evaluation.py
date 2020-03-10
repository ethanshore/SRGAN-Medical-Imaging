import sklearn
from linear-interp import img as im_test
from linear-interp import image as im_true

# peak signal to noise ratio
skimage.measure.compare_psnr(im_true, im_test, data_range=None)

# SSIM
skimage.measure.compare_ssim(X, Y, win_size=None, gradient=False, data_range=None, multichannel=False, gaussian_weights=False, full=False, **kwargs)skimage.measure.compare_ssim(X, Y, win_size=None, gradient=False, data_range=None, multichannel=False, gaussian_weights=False, full=False, **kwargs)