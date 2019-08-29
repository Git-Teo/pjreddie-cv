import convolution as cnv
import cv2
import numpy as np
import math

img = cv2.imread('images/dog.jpeg')
sbimg = cnv.sobel_image(img)
norm_sbimg = cnv.normalize_image(sbimg[0])
# cv2.imwrite('images/sobeldog.jpeg', norm_sbimg * 255)
img = cv2.imread('images/sobeldog.jpeg')
# cv2.imshow('sobel dog', img)
# cv2.waitKey()
hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
print(hsvimg[:,:,2])
hsvimg[:,:,0] = np.multiply(np.divide(sbimg[1],(2*math.pi), out=np.zeros_like(yimg), where=ximg!=0) * 255, hsvimg[:,:,0])
print(hsvimg[:3,:3])
cv2.imshow('magnitude normalized', hsvimg)
hsvimg[:,:,1] = sbimg[0]
print(hsvimg[:3,:3])
print(np.divide(sbimg[1],(2*math.pi)) * 255)
cv2.imshow('magnitude normalized2', hsvimg)

cv2.waitKey()
