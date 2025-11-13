import cv2 as cv
import numpy as np
import math

def gradient_intensity(image):
    sobel_x = np.array([
        [-1,0,1],
        [-2,0,2],
        [-1,0,1]
    ])
    sobel_y = np.array([
        [-1,-2,-1],
        [0,0,0],
        [1,2,1]
    ])

    padded = np.pad(image, pad_width=1, mode='edge')
    rows, cols = image.shape
    grad_x = np.zeros_like(image, dtype=np.float32)
    grad_y = np.zeros_like(image, dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            roi = padded[i:i+3, j:j+3]
            grad_x[i, j] = np.sum(roi * sobel_x)
            grad_y[i, j] = np.sum(roi * sobel_y)

    vector_length = np.sqrt(grad_x**2+grad_y**2)
    difference_way = np.arctan2(grad_y, grad_x)
    return vector_length, difference_way

if __name__ == "__main__":
    image = cv.imread("lab4/input/gelenzhik.jpg", cv.IMREAD_GRAYSCALE)
    sigma, size = 1,3
    filtered_image = cv.GaussianBlur(image, (size, size), sigma)
    vector_length, difference_way = gradient_intensity(filtered_image)
    

    cv.namedWindow('Original', cv.WINDOW_FREERATIO)
    cv.namedWindow('Gaussian Filtered', cv.WINDOW_FREERATIO)
    cv.imshow("Original", image)
    cv.imshow("Gaussian Filtered", filtered_image)
    cv.waitKey(0)
    cv.destroyAllWindows()