import numpy as np
import cv2
import matplotlib.pyplot as plt

def lbp(image):
    lbp_image = np.zeros_like(image)
    height, width = image.shape
    directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            center_pixel = image[i, j]
            binary_val = []
            for dx, dy in directions:
                if image[i + dx, j + dy] >= center_pixel:
                    binary_val.append(1)
                else:
                    binary_val.append(0)
            lbp_val = sum([2 ** i for i, val in enumerate(binary_val) if val])
            lbp_image[i, j] = lbp_val

    return lbp_image


image = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)
lbp_result = lbp(image)

plt.imshow(lbp_result, cmap='gray')
plt.title('lbp')
plt.show()
