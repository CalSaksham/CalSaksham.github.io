import matplotlib.pyplot as plt

from align_image_code import align_images
import scipy.signal
import numpy as np
import cv2

# First load images

# high sf
im1 = plt.imread('./Campanile Hi-Res.jpg') / 255.

# low sf
im2 = plt.imread('./Eiffel Tower cropped.jpg') / 255.


im3 = plt.imread('./sachin.jpg') / 255.
im4 = plt.imread('./virat.jpg') / 255.
im5 = plt.imread('./DerekPicture.jpg') / 255.
im6 = plt.imread('./nutmeg.jpg') / 255.

# Next align images (this code is provided, but may be improved)
im1_aligned, im2_aligned = align_images(im1, im2)
im1_aligned = im1_aligned.mean(axis=2)
im2_aligned = im2_aligned.mean(axis=2)

## You will provide the code below. Sigma1 and sigma2 are arbitrary 
## cutoff values for the high and low frequencies

def hybrid_image(img1, img2, sigma1, sigma2):
    low_kernel = np.dot(cv2.getGaussianKernel(sigma2*6+1,sigma2), cv2.getGaussianKernel(sigma2*6+1,sigma2).T)
    high_kernel = np.dot(cv2.getGaussianKernel(sigma1*6+1,sigma1), cv2.getGaussianKernel(sigma1*6+1,sigma1).T)


    low = scipy.signal.convolve2d(img2, low_kernel, mode='same')
    high = img1 - scipy.signal.convolve2d(img1, high_kernel, mode='same')

    out = np.clip(low + high, 0, 1)
    return low, high, out

gaussian_kernel = np.dot(cv2.getGaussianKernel(9,1), cv2.getGaussianKernel(9,1).T)


sigma1 = 6
sigma2 = 10

low, high, hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)

plt.figure()
plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(im1_aligned)))))
plt.title('input im1')
plt.figure()
plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(im2_aligned)))))
plt.title('input im2')
plt.figure()
plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(low)))))
plt.title('low_res')
plt.figure()
plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(high)))))
plt.title('high_res')

plt.figure()
plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(hybrid)))))
plt.title('hybrid FFT')


plt.figure()
plt.imshow(hybrid, cmap='gray')
plt.title('hybrid')
plt.show()