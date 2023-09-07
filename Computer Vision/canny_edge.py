import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def normalise(image):
    norm_image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    return norm_image.astype(np.uint8)

def gaussian(size, sigma):
    half = size // 2
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            x = i - half
            y = j - half
            kernel[i][j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

def sobel(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) 
    derv_x = cv.filter2D(image, -1, sobel_x)
    derv_y = cv.filter2D(image, -1, sobel_y)
    magnitude = np.sqrt(derv_x**2 + derv_y**2)
    angle = np.arctan2(derv_y, derv_x) * 180 / np.pi
    angle[angle < 0] += 180
    
    for i in range(1, magnitude.shape[0]-1):
        for j in range(1, magnitude.shape[1]-1):
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                value_to_compare = np.maximum(magnitude[i, j-1], magnitude[i, j+1])
            elif 22.5 <= angle[i,j] < 67.5:
                value_to_compare = np.maximum(magnitude[i-1, j+1], magnitude[i+1, j-1])
            elif 67.5 <= angle[i,j] < 112.5:
                value_to_compare = np.maximum(magnitude[i-1, j], magnitude[i+1, j])
            else:
                value_to_compare = np.maximum(magnitude[i-1, j-1], magnitude[i+1, j+1])
            
            if magnitude[i,j] < value_to_compare:
                magnitude[i,j] = 0

    return magnitude

def laplacian(image):
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    return cv.filter2D(image, -1, kernel)

def canny_edge(laplacian):
    laplacian = laplacian.astype(np.float64)
    height, width = laplacian.shape
    zero_crossing = np.zeros((height, width), dtype=np.uint8)

    for i in range(1, height-1):
        for j in range(1, width-1):
            if laplacian[i, j] * laplacian[i+1, j] < 0 or laplacian[i, j] * laplacian[i, j+1] < 0:
                zero_crossing[i, j] = 255

    return zero_crossing

image = cv.imread('image3.jpg', cv.IMREAD_GRAYSCALE)
image = normalise(image)

kernel = gaussian(15, 1.5)
new_image = cv.filter2D(image, -1, kernel)

plt.imshow(new_image, cmap='gray')
plt.title('After Gaussian')
plt.show()

new_image = normalise(new_image)
new_image2 = sobel(new_image)


plt.imshow(new_image2, cmap='gray')
plt.title('After Sobel')
plt.show()

new_image2 = normalise(new_image2)
new_image3 = laplacian(new_image2)
plt.imshow(new_image3, cmap='gray')
plt.title('After Laplacian')
plt.show()

new_image3 = normalise(new_image3)

new_image4 = canny_edge(new_image3)
plt.imshow(new_image4, cmap='gray')
plt.title('Result')
plt.show()

import cv2 as cv
import matplotlib.pyplot as plt
image = cv.imread('image3.jpg', cv.IMREAD_GRAYSCALE)
edges = cv.Canny(image, 50, 150)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detection using OpenCV')
plt.show()

