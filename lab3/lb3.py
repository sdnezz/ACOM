import cv2 as cv
import numpy as np
import math

def gaussian_kernel(size, sigma):
    center = size//2 # потмоу что с 0 массив индексируется, центр для 3 - 2, для 5 - 3, но в индексах 1 и 2
    kernel = np.zeros((size, size))
    print(f"Центр (Мат. ожидание): a={center}, b={center}")

    for i in range(size):
        for j in range(size):
            x=i-center
            y=j-center
            kernel[i, j] = (1/(2*math.pi*sigma**2))*math.exp(-(x**2+y**2)/(2 * sigma**2))

    kernel /= kernel.sum()
    kernel = kernel / kernel.sum()
    print(kernel.sum())
    return kernel

def apply_gaussian_filter(image, kernel):
    kernel_size = kernel.shape[0]
    pad = kernel_size // 2

    padded_image = cv.copyMakeBorder(image, pad, pad, pad, pad, cv.BORDER_REPLICATE)

    will_filtered_empty_image = np.zeros_like(image, dtype=np.float64)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_image[i:i+kernel_size, j:j+kernel_size]
            will_filtered_empty_image[i, j] = np.sum(window * kernel)

    return np.clip(will_filtered_empty_image, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    for size in [3, 7]:
        for sigma in [1, 3]:
            kernel = gaussian_kernel(size, sigma=sigma)
            print(f"Гауссова матрица для размера {size}x{size} и сигмы {sigma}:\n", kernel)

            image = cv.imread("lab1/input/tytuta.jpg", cv.IMREAD_GRAYSCALE)
            filtered_image = apply_gaussian_filter(image, kernel)

            cv.imshow("Original", image)
            cv.imshow("Gaussian Filtered", filtered_image)
        cv.waitKey(0)
        cv.destroyAllWindows()