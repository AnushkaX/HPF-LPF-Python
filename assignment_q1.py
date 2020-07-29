from _json import make_scanner

import cv2
import numpy as np
from matplotlib import pyplot as plt

# step 1
img = cv2.imread("", 0)

# step 2
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

# step 3
rows, cols = img.shape
crow, ccol = int(rows / 2), int(cols / 2)

# Smoothing lpf
# create a mask first , center square is 1, remaining all zeros
mask_lpf = np.zeros((rows, cols, 2), np.uint8)
mask_lpf[crow - 30:crow + 30, ccol - 30: ccol + 30] = 1

# hpf
mask_hpf = np.ones((rows, cols, 2), np.uint8)
mask_hpf[crow - 30:crow + 30, ccol - 30: ccol + 30] = 0

# apply mask and inverse dft
fshift_lpf = dft_shift * mask_lpf
fshift_hpf = dft_shift * mask_hpf

# step 4
f_ishift_lpf = np.fft.ifftshift(fshift_lpf)
img_back_lpf = cv2.idft(f_ishift_lpf)

f_ishift_hpf = np.fft.ifftshift(fshift_hpf)
img_back_hpf = cv2.idft(f_ishift_hpf)

img_back_lpf = cv2.magnitude(img_back_lpf[:, :, 0], img_back_lpf[:, :, 1])
img_back_hpf = cv2.magnitude(img_back_hpf[:, :, 0], img_back_hpf[:, :, 1])

# step 5
plt.subplot(221), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum')
plt.subplot(223), plt.imshow(img_back_lpf, cmap='gray')
plt.title('Lowpass Image'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(img_back_hpf, cmap='gray')
plt.title('Highpass Image'), plt.xticks([]), plt.yticks([])

plt.show()
cv2.waitKey()
