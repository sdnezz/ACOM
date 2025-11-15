import cv2 as cv
import numpy as np
import math

def gradient_intensity(image):
    sobel_x = np.array([
        [-1,0,1],
        [-2,0,2],
        [-1,0,1]
    ], dtype=np.float32)
    sobel_y = np.array([
        [-1,-2,-1],
        [0,0,0],
        [1,2,1]
    ], dtype=np.float32)

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
    vector_angle = np.arctan2(grad_y, grad_x)
    return vector_length, vector_angle

def non_maximum_suppression(vec_lenght, vec_angle):
    suppressed = np.zeros_like(vec_lenght, dtype=np.float32)
    angle = vec_angle * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, vec_lenght.shape[0] - 1):
        for j in range(1, vec_angle.shape[1] - 1):
            q = 255
            r = 255
            #сравнение длины градиента каждого пикселя с его соседями
            #(если он не максимальный, то обнуляем для четких)
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = vec_lenght[i, j + 1] #правый сосед
                r = vec_lenght[i, j - 1] #левый сосед
            elif 22.5 <= angle[i, j] < 67.5:
                q = vec_lenght[i + 1, j - 1] #нижний-левый
                r = vec_lenght[i - 1, j + 1] #верхний-правый
            elif 67.5 <= angle[i, j] < 112.5:
                q = vec_lenght[i + 1, j] #нижний сосед
                r = vec_lenght[i - 1, j] #верхний сосед
            elif 112.5 <= angle[i, j] < 157.5:
                q = vec_lenght[i - 1, j - 1] #верзний-правый
                r = vec_lenght[i + 1, j + 1] #нижний-правый

            if vec_lenght[i, j] >= q and vec_lenght[i, j] >= r:
                suppressed[i, j] = vec_lenght[i, j]
            else:
                suppressed[i, j] = 0

    max_grad = np.max(suppressed)
    low_level = max_grad // 20
    high_level = max_grad // 2.5
    print(low_level, high_level)
    return suppressed, low_level, high_level

def double_threshold(suppressed_image, low_level, high_level):
    strong = suppressed_image >= high_level
    weak = (suppressed_image >= low_level) & (suppressed_image < high_level)

    ways = np.zeros_like(suppressed_image, dtype=np.uint8)
    ways[strong] = 255
    ways[weak] = 75
    return ways

def hysteresis(ways):
    rows, cols = ways.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if ways[i, j] == 75:
                if np.any(ways[i-1 : i+2, j-1 : j+2] == 255):
                    ways[i, j] = 255
                else:
                    ways[i, j] = 0

    return ways

if __name__ == "__main__":
    image = cv.imread("lab4/input/gelenzhik.jpg")
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    sigma, size = 1,3
    filtered_image = cv.GaussianBlur(image, (size, size), sigma)
    vector_length, vector_angle = gradient_intensity(filtered_image)
    suppressed_image, low_lvl, high_lvl = non_maximum_suppression(vector_length, vector_angle)
    ways = double_threshold(suppressed_image, low_lvl, high_lvl)
    edges = hysteresis(ways)

    cv.imwrite("lab4/output/Canny_detect-gelenzhik.jpg", edges)
    cv.imwrite("lab4/output/cv_canny-gelenzhik.jpg", cv.Canny(image, 400, 500))
    image_canny = cv.imread("lab4/output/Canny_detect-gelenzhik.jpg")
    image_canny_cv = cv.imread("lab4/output/cv_canny-gelenzhik.jpg")
    cv.namedWindow('Original', cv.WINDOW_FREERATIO)
    cv.namedWindow('Canny', cv.WINDOW_FREERATIO)
    cv.namedWindow("OpenCV", cv.WINDOW_FREERATIO)
    cv.imshow("Original", image)
    cv.imshow("Canny", image_canny)
    cv.imshow("OpenCV", image_canny_cv)
    cv.waitKey(0)
    cv.destroyAllWindows()