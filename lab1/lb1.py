import cv2 as cv
print(cv.__version__)

tuta_image = cv.imread("input/tytuta.jpg")
cv.imshow('output', tuta_image)
cv.waitKey(0)