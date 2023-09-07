import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
def histogram(image,k):
    hist=[0]*k
    for i in image.flatten():
        hist[i]+=1
    return hist

def cdf_calc(image,k):
    hist=histogram(image,k)
    cdf=[0]*k
    cdf[0]=hist[0]
    for i in range(1,k):
        cdf[i]=cdf[i-1]+hist[i]
    return cdf

def histogram_equalisation(image,k):
    cdf=cdf_calc(image,k)
    height,width=image.shape
    size=height*width
    new_image=[[0 for _ in range(width)] for _ in range(height)]
    for y in range(height):
        for x in range(width):
            pixel=image[y][x]
            new_pixel=int(cdf[pixel]/size*k)
            new_image[y][x]=new_pixel
    return new_image

image=cv.imread('image6.jpeg',cv.IMREAD_GRAYSCALE)
new_image=histogram_equalisation(image,256)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(new_image,cmap='gray')
plt.show()