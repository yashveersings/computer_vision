import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def negative(image):
    height, width = image.shape
    new_image = np.zeros_like(image)
    new_image = 255 - image
    return new_image

def log(image):
    c = 255 / np.log(1 + np.max(image))
    new_image = c * np.log(1 + image)
    return new_image

def gamma(image, gamma):
    new_image = 255 * (image / 255) ** gamma
    return new_image

image = cv.imread('image6.jpeg', cv.IMREAD_GRAYSCALE)
new_image = negative(image)

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(new_image, cmap='gray')

new_image2 = log(image)
plt.subplot(1, 3, 2)
plt.imshow(new_image2, cmap='gray')

new_image3 = gamma(image, 0.5)
plt.subplot(1, 3, 3)
plt.imshow(new_image3, cmap='gray')

# Convert NumPy arrays to OpenCV images
new_image_cv = cv.cvtColor(new_image.astype(np.uint8), cv.COLOR_BGR2RGB)
new_image2_cv = cv.cvtColor(new_image2.astype(np.uint8), cv.COLOR_BGR2RGB)
new_image3_cv = cv.cvtColor(new_image3.astype(np.uint8), cv.COLOR_BGR2RGB)

cv.imshow('a', new_image_cv)
cv.waitKey()

cv.imshow('b', new_image2_cv)
cv.waitKey()

cv.imshow('c', new_image3_cv)
cv.waitKey()

cv.destroyAllWindows()  # Close all OpenCV windows when done
