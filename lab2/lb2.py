import cv2
import numpy as np

camera = cv2.VideoCapture(0)

# task 1
while camera.isOpened():
    ok, img= camera.read()
    if not(ok):
        break

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    cv2.imshow("HSV", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()

