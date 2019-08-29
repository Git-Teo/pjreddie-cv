import cv2
import numpy as np
import math

img = cv2.imread('images/dog.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
filtered_img = np.zeros((img.shape[0],img.shape[1]), np.uint8)
cv2.imshow('grayscale_image', img)

filter = np.array([[0, 0, 0],
                   [0, 1, 0],
                   [0, 0, 0]])

# filter = np.array([[-1, 0, 1],
#                    [-2, 0, 2],
#                    [-1, 0, 1]])

filter = np.array([[0, -1, 0],
                   [-1, 4, -1],
                   [0, -1, 0]])

# img = np.array([[0, -1, 0],
#                [-1, 4, -1],
#                [0, -1, 0]])
origin = [1,1]

# print(np.sum(np.multiply(filter, test)))
def pad_image(img, l_x, r_x, l_y, r_y):
    h, w = img.shape[0:2]
    # print(l_x, r_x, l_y, r_y)
    # print(h, w)
    padded_img = np.zeros((h+l_y+r_y, w+l_x+r_x))
    # print(padded_img)
    i_l_x, i_r_x = (l_x, w+l_x)
    i_l_y, i_r_y = (l_y, h+l_y)
    # print(i_l_x, i_r_x, i_l_y, i_r_y)
    padded_img[i_l_y:i_r_y, i_l_x:i_r_x] = img
    return [padded_img, i_l_x, i_r_x, i_l_y, i_r_y]

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

def convolution(filter, origin, img):
    i_h, i_w = img.shape
    f_h, f_w = filter.shape
    # print(i_h, i_w, f_h, f_w)
    res_img = np.zeros((i_h, i_w, 1), np.uint8)
    img, i_l_x, i_r_x, i_l_y, i_r_y = pad_image(img, origin[0], f_w-origin[0]-1, origin[1], f_h-origin[1]-1)
    for h, row in enumerate(img):
        if h >= i_l_y and h < i_r_y:
            p = 0
            for w, col in enumerate(row):
                if w >= i_l_x and w < i_r_x:
                    # print(np.multiply(filter, img[h-i_l_y:h+i_l_y+1, w-i_l_x:w+i_l_x+1]))
                    # print(np.sum(np.multiply(filter, img[h-i_l_y:h+i_l_y+1, w-i_l_x:w+i_l_x+1])))
                    # res_img[h-i_l_y][w-i_l_x] = np.sum(np.multiply(filter, img[h-i_l_y:h+i_l_y+1, w-i_l_x:w+i_l_x+1]))/(f_h*f_w)
                    res_img[h-i_l_y][w-i_l_x] = max(0, np.sum(np.multiply(filter, img[h-i_l_y:h+i_l_y+1, w-i_l_x:w+i_l_x+1])))
                    if p:
                        print('image')
                        print(img[h-i_l_y:h+i_l_y+1, w-i_l_x:w+i_l_x+1])
                        print('res')
                        print(np.sum(np.multiply(filter, img[h-i_l_y:h+i_l_y+1, w-i_l_x:w+i_l_x+1])))
                        print(np.multiply(filter, img[h-i_l_y:h+i_l_y+1, w-i_l_x:w+i_l_x+1]))
                        print(res_img)
                        print()
    return res_img

def gaussian(x, sigma):
    return math.exp(-(x**2)/(2*(sigma**2))) / (math.sqrt(2*math.pi*sigma**2))

def gaussian_2d(x, y, sigma):
    return math.exp(-(((x**2) + (y**2))/(2*(sigma**2)))) * (1 / (2*math.pi*(sigma**2)))


res = convolution(filter, origin, img)
cv2.imshow('image', res)
cv2.waitKey()
