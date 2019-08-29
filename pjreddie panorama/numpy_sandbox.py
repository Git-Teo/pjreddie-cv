import cv2
import numpy as np

filter = np.array([[0, -1, 0],
                   [-1, 4, -1],
                   [0, -1, 0]])

img = np.array([
            [[0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]],

            [[-1, 4, -1],
            [0, -1, 0],
            [0, 0, 0]],

            [[0, -1, 0],
            [-1, 1, -1],
            [0, -1, 0]]])

# img = np.array([[-1, 4, -1],
#                    [0, -1, 0],
#                    [0, 0, 0]])
print(img + 1)
print(np.multiply([255],[255]))
# print(np.dot(filter, img))
print(np.multiply(filter, img))
print(np.sum(np.multiply(filter, img)))
