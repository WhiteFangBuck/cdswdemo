%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow

import cv2

img = cv2.imread('leo.png')
plt.imshow(img)


# convert image to RGB color for matplotlib
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# show image with matplotlib
plt.imshow(img)

# convert image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#if gray_img == None: 
#  raise Exception("could not load image !")
plt.imshow(gray_img)
print gray_img.shape


## convert image to grayscale
#color_img = cv2.cvtColor(gray_img, cv2.COLOR_RGB2GRAY)
#if color_img == None: 
#  
#  raise Exception("could not load image !")
#plt.imshow(color_img)

# threshold for grayscale image
_, threshold_img = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY)

threshold_img = cv2.cvtColor(threshold_img, cv2.COLOR_GRAY2RGB)
plt.figure(figsize=(10,10))
plt.imshow(threshold_img)

# preproccess with blurring, with 5x5 kernel
img_blur_small = cv2.GaussianBlur(img, (5,5), 0)
cv2.imwrite('output/oy-gaussian-blur-5.jpg', img_blur_small)
plt.imshow(cv2.cvtColor(img_blur_small, cv2.COLOR_BGR2RGB))

img_blur_large = cv2.GaussianBlur(img, (15,15), 0)
cv2.imwrite('output/oy-gaussian-blur-15.jpg', img_blur_large)
plt.imshow(cv2.cvtColor(img_blur_large, cv2.COLOR_BGR2RGB))
#img2 = img[:,:,::-1]
#plt.imshow(img2)
#img=mpimg.imread('11dogs-rising2-master495.jpg')

# using adaptive threshold instead of global
adaptive_thresh = cv2.adaptiveThreshold(gray_img,255,\
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                         cv2.THRESH_BINARY,11,2)
plt.imshow(cv2.cvtColor(adaptive_thresh, cv2.COLOR_GRAY2RGB))
