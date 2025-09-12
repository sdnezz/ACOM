import cv2 as cv
print(cv.__version__)

#TASK 1
# tuta_image = cv.imread("input/tytuta.jpg")
# cv.namedWindow('Display window', cv.WINDOW_FREERATIO)
# cv.imshow('Display window', tuta_image)
# cv.waitKey(0)
# cv.destroyAllWindows()

#TASK 2

tuta_image_pixeled = cv.imread("input/tytuta.jpg", cv.IMREAD_REDUCED_COLOR_8)
tuta_image_negative = cv.imread("input/tytuta.bmp", cv.IMREAD_COLOR_RGB)
tuta_image_wb = cv.imread("input/tytuta.png", cv.IMREAD_GRAYSCALE)
cv.namedWindow('free window ratio', cv.WINDOW_FREERATIO)
cv.namedWindow('normal window', cv.WINDOW_NORMAL)
cv.namedWindow('autosize window', cv.WINDOW_AUTOSIZE)
cv.imshow('free window ratio', tuta_image_pixeled)
cv.imshow('autosize window', tuta_image_negative)
cv.imshow('normal window', tuta_image_wb)
cv.waitKey(0)
cv.destroyAllWindows()
