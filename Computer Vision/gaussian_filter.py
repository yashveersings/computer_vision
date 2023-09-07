import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(size,sigma):
    half_size=size//2
    kernel=[[0]*size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            x=i-half_size
            y=j-half_size
            kernel[i][j] = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return np.array(kernel)/np.sum(kernel)
image=cv.imread('image6.jpeg',cv.IMREAD_GRAYSCALE)
kernel1 = gaussian_kernel(100, 5.0)
new_image=cv.filter2D(src=image,ddepth=-1,kernel=kernel1)
plt.figure(figsize=(12,6))
plt.subplot(1,2,2)
plt.imshow(new_image)

cv.imshow('abc',new_image)
cv.waitKey()



