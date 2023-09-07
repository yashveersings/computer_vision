def gradient(image):
    sobel_x=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_y=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    x_derivative=cv.filter2D(image,-1,sobel_x)
    y_derivative=cv.filter2D(image,-1,sobel_y)
    magnitude=np.sqrt(x_derivative**2+y_derivative**2)
    return magnitude

image=cv.imread('image6.jpeg',cv.IMREAD_GRAYSCALE)
new_image=gradient(image)
plt.imshow(image,cmap='gray')
plt.show()

max=np.max(new_image)
min=np.min(new_image)
new_image=(new_image-min)/(max-min)

plt.imshow(new_image,cmap='gray')
plt.show()