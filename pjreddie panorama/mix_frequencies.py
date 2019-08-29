import convolution as cnv
import cv2
import numpy as np

img = cv2.imread('images/ron.jpeg')

n = 30
if n % 2 == 0:
    n += 1

kernel = cnv.make_gaussian_kernel(n, 2)
origin = [(n-1)//2,(n-1)//2]

lfreq_ron = cnv.convolve_image(kernel, origin, img, 1)

hfreq_ron = img - lfreq_ron

img = cv2.imread('images/dumbledore.jpeg')

lfreq_dd = cnv.convolve_image(kernel, origin, img, 1)

cv2.imshow('rondore', lfreq_dd+hfreq_ron)

# dd_lfreq = convolve_image(kernel, origin, img, 1)

cv2.imshow('ron high f', lfreq_dd)
cv2.imshow('ron low f', lfreq_ron)

cv2.waitKey()
