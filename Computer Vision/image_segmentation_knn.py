import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
image=cv.imread('image6.jpeg')
x=image.reshape(-1,3)
kmeans=KMeans(n_clusters=1,n_init=10)
kmeans.fit(x)
segmented_image=kmeans.cluster_centers_[kmeans.labels_]
segmented_image=segmented_image.reshape(image.shape)
plt.imshow(segmented_image/255)
plt.show()