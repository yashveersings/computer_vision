import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def box_filter(size):
    kernel=[[1]*size for _ in range(size)]
    return kernel/np.sum(kernel)
image=cv.imread('image6.jpeg',cv.IMREAD_GRAYSCALE)
kernel1 = box_filter(5)
new_image=cv.filter2D(src=image,ddepth=-1,kernel=kernel1)
plt.figure(figsize=(12,6))
plt.subplot(1,2,2)
plt.imshow(new_image)

cv.imshow('abc',new_image)
cv.waitKey()



