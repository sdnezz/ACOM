import cv2 as cv
import numpy as np

def gaussian_kernel(size, sigma):
    center = int(np.ceil(size / 2) - 1) # потмоу что с 0 массив индексируется, центр для 3 - 2, для 5 - 3, но в индексах 1 и 2
    kernel = np.zeros((size, size), dtype=np.float64)
    print(f"Центр (Мат. ожидание): a={center}, b={center}")
    for x in range(size):
        for y in range(size):
            kernel[x, y] = (1 / (2 * np.pi * sigma**2)) * np.exp(-((x-center)**2 + (y-center)**2)) / (2 * sigma**2)
    
    kernel /= kernel.sum()
    kernel = kernel / kernel.sum()
    print(kernel.sum())
    return kernel

if __name__ == "__main__":
    for size in [3, 5, 7]:
        kernel = gaussian_kernel(size, sigma=1.0)
        print(f"Гауссова матрица для размера {size}x{size}:\n", kernel)