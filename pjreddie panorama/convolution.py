import cv2
import numpy as np
import math

def pad_image(img, kernel, origin):
    f_h, f_w = kernel.shape
    l_x, r_x, l_y, r_y = origin[0], f_w-origin[0]-1, origin[1], f_h-origin[1]-1
    if len(img.shape) == 3:
        h, w, c = img.shape[0:3]
        padded_img = np.zeros((h+l_y+r_y, w+l_x+r_x, c))
    else:
        h, w = img.shape[0:2]
        padded_img = np.zeros((h+l_y+r_y, w+l_x+r_x))
    padded_img[l_y:h+l_y, l_x:w+l_x] = img
    frame = [l_x, l_y, w+l_x, h+l_y]
    return [padded_img, frame]

def convolve_image(kernel, origin, img, preserve):
    if (len(img.shape) == 3):
        i_h, i_w, i_c = img.shape
        res_img = np.zeros((i_h, i_w, i_c), np.uint32) if preserve else np.zeros((i_h, i_w), np.uint32)
    else:
        i_h, i_w = img.shape
        i_c = 0
        res_img = np.zeros((i_h, i_w), np.uint32)
    f_h, f_w = kernel.shape
    # print(i_c, i_h, i_w, f_h, f_w)
    # print(res_img)
    img, frame = pad_image(img, kernel, origin)
    for h, row in enumerate(img):
        if h >= frame[1] and h < frame[3]:
            for w, col in enumerate(row):
                if w >= frame[0] and w < frame[2]:
                    if i_c:
                        for c in range(0, i_c):
                            if preserve:
                                res_img[h-frame[1]][w-frame[0]][c] = max(0, min(255, np.sum(np.multiply(kernel, img[h-frame[1]:h+frame[1]+1, w-frame[0]:w+frame[0]+1, c]))))
                            else:
                                res_img[h-frame[1]][w-frame[0]] = max(0, min(255, np.sum(np.multiply(kernel, img[h-frame[1]:h+frame[1]+1, w-frame[0]:w+frame[0]+1, c]))))
                    else:
                        res_img[h-frame[1]][w-frame[0]] = max(0, min(255, np.sum(np.multiply(kernel, img[h-frame[1]:h+frame[1]+1, w-frame[0]:w+frame[0]+1]))))
    return res_img

def make_gaussian_kernel(d, sigma):
    if d % 2 == 0:
        d += 1
    o = (d-1) / 2
    f = np.zeros((d,d), np.double)
    for x, y in np.ndindex((d,d)):
        # print(gaussian_2d(x-o, y-o, sigma))
        f[x][y] = gaussian_2d(x-o, y-o, sigma)
    return f

def make_highpass_kernel():
    return np.array([[0, -1, 0],[-1, 4, -1],[0, -1, 0]])

def make_sharper_kernel():
    return np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])

def make_emboss_kernel():
    return np.array([[-2, -1, 0],[-1, 1, 1],[0, 1, 2]])

def make_emboss_kernel():
    return np.array([[-2, -1, 0],[-1, 1, 1],[0, 1, 2]])

def make_sobel_x_kernel():
    return np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])

def make_sobel_y_kernel():
    return np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])

def gaussian_2d(x, y, sigma):
    return math.exp(-(((x**2) + (y**2))/(2*(sigma**2)))) * (1 / (2*math.pi*(sigma**2)))

def sobel_image(img):
    yk = make_sobel_y_kernel()
    xk = make_sobel_x_kernel()
    yimg = convolve_image(yk, [1,1], img, 0)
    ximg = convolve_image(xk, [1,1], img, 0)
    xsimg = np.multiply(ximg, ximg)
    ysimg = np.multiply(yimg, yimg)
    sumimg = xsimg + ysimg
    mag = np.sqrt(sumimg)
    dir = np.arctan(np.divide(yimg, ximg, out=np.zeros_like(yimg), where=ximg!=0))
    return [mag, dir]

def normalize_image(img):
    min = np.min(img)
    r = np.amax(img) - min
    return np.zeros(img.shape) if r == 0 else np.divide((img - min), r)

# img = cv2.imread('images/dog.jpeg')
# # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# n = 3
# if n % 2 == 0:
#     n += 1
#
# kernel = make_gaussian_kernel(n, 1)
# origin = [(n-1)//2,(n-1)//2]
#
# blur = convolve_image(kernel, origin, img, 1)
#
# img = cv2.imread('images/dog.jpeg')
#
# n = 30
# if n % 2 == 0:
#     n += 1
#
# kernel = make_gaussian_kernel(n, 2)
# origin = [(n-1)//2,(n-1)//2]
#
# lfreq = convolve_image(kernel, origin, img, 1)
#
# n = 30
# if n % 2 == 0:
#     n += 1
#
# kernel = make_gaussian_kernel(n, 2)
# origin = [(n-1)//2,(n-1)//2]
#
# lfreq_2 = convolve_image(kernel, origin, img, 1)
#
# lf_dog = cv2.imread('images/low-frequency-dog.jpeg')
# cv2.imshow('diff', lf_dog - lfreq)
# cv2.imshow('diff_2', lf_dog - lfreq_2)
#
# hf_dog = cv2.imread('images/high-frequency-dog.jpeg')
# hfreq = img - lfreq
# # cv2.imshow('diff', hf_dog - hfreq)
# # img = cv2.imread('images/dumbledore.jpeg')
#
# # dd_lfreq = convolve_image(kernel, origin, img, 1)
#
# cv2.imshow('ron high f', hfreq)
# cv2.imshow('ron low f', lfreq)
#
# cv2.waitKey()

# origin = [1,1]
# highpass = convolve_image(make_highpass_kernel(), origin, img, 1)
# sharp = convolve_image(make_sharper_kernel(), origin, img, 1)
# emboss = convolve_image(make_emboss_kernel(), origin, img, 1)
# cv2.imshow('highpass', highpass)
# cv2.imshow('sharp', sharp)
# cv2.imshow('emboss', emboss)
# cv2.waitKey()


