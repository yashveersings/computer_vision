import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def histogram(image, k):
    hist = [0] * k
    for i in image.flatten():
        hist[i] += 1
    return hist

def cdf_calc(image, k):
    hist = histogram(image, k)
    cdf = [0] * k
    cdf[0] = hist[0]
    for i in range(1, k):
        cdf[i] = cdf[i - 1] + hist[i]
    return cdf

def histogram_matching(input_image, output_image, k):
    cdf_input = cdf_calc(input_image, k)
    cdf_ref = cdf_calc(output_image, k)
    height, width = input_image.shape
    new_image = np.zeros_like(input_image, dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            pixel = input_image[y][x]
            new_pixel = int(cdf_ref[pixel] / cdf_input[pixel] * (k - 1))
            new_pixel = max(0, min(new_pixel, 255))  # Clip the values to [0, 255]
            new_image[y][x] = new_pixel
    return new_image

image = cv.imread('image.jpg', cv.IMREAD_GRAYSCALE)
image2 = cv.imread('image4.jpg', cv.IMREAD_GRAYSCALE)
new_image = histogram_matching(image, image2, 256)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(new_image, cmap='gray')
plt.show()

cv.imshow('image', new_image)
cv.waitKey()
