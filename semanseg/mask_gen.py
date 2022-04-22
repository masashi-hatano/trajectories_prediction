import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('output/result_seg_originalsize.jpg')

for h in range(img.shape[0]):
    for w in range(img.shape[1]):
        if img[h][w][0] != 127 and img[h][w][1] != 63 and img[h][w][2] != 128:
            img[h][w] = np.array([0,0,0])

cv2.imwrite('./mask_road.jpg', img)
plt.imshow(img)
plt.show()
