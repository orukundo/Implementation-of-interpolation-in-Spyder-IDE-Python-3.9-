# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 20:20:40 2022

@author: oruku
"""
# Import necessary libraries
import cv2
import matplotlib.pyplot as plt

# Select the scaling/zooming ratio
ratio = 4

# Read 2D/grayscale image (non-interpolated)
im = cv2.imread('C:/Users/oruku/PYTHON-CODE/adenovirusdataset/images/p_img1_4.tiff')
# Get dimensions of 2D image
old_h, old_w, c = im.shape
new_h = ratio * old_h
new_w = ratio * old_w
# Nearest neighbor interpolation
im_nearest = cv2.resize(im, (new_h, new_w), interpolation = cv2.INTER_NEAREST)
# Bilinear interpolation
im_linear = cv2.resize(im, (new_h, new_w), interpolation = cv2.INTER_LINEAR)
# Bicubic interpolation
im_cubic = cv2.resize(im, (new_h, new_w), interpolation = cv2.INTER_CUBIC)
# Lanczos-kernel interpolation
im_lanczos = cv2.resize(im, (new_h, new_w), interpolation = cv2.INTER_LANCZOS4)

# Display interpolated images
plt.figure(figsize = (24, 24))
plt.subplot(221)
img = plt.imshow(im_nearest)
plt.subplot(222)
plt.imshow(im_linear)
plt.subplot(223)
plt.imshow(im_cubic)
plt.subplot(224)
plt.imshow(im_lanczos)
plt.show()