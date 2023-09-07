import numpy as np
import cv2 as cv
import cv2 as cv
import matplotlib.pyplot as plt

image = cv.imread('image3.jpg', cv.IMREAD_GRAYSCALE)
height, width = image.shape
new_image = [[0 for _ in range(width)] for _ in range(height)]


def binary_image(image):
    for y in range(height):
        for x in range(width):
            pixel = image[y][x]
            if pixel >= 120:
                new_pixel = 0
            else:
                new_pixel = 255
            new_image[y][x] = new_pixel
    return new_image


new_image = binary_image(image)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(new_image, cmap='gray')
plt.show()

