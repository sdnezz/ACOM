import cv2 as cv
print(cv.__version__)

tuta_image = cv.imread("input/tytuta.jpg")
cv.namedWindow('Display window', cv.WINDOW_FREERATIO)
cv.imshow('Display window', tuta_image)
cv.waitKey(0)
cv.destroyAllWindows()