import cv2 as cv
import numpy as np

camera = cv.VideoCapture(0)

def task1():
    while camera.isOpened():
        ok, img= camera.read()
        if not(ok):
            break

        img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        cv.imshow("HSV", img)

        if cv.waitKey(1) & 0xFF == 27:
            break

    cv.destroyAllWindows()

def task2():
    while camera.isOpened():
        ok, img = camera.read()
        if not(ok):
            break

        HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        low_color_red = np.array((100, 90, 100))
        high_color_red = np.array((140, 190, 190))

        treshold = cv.inRange(HSV, low_color_red, high_color_red)

        cv.imshow("Default", img)
        cv.imshow("only Red", treshold)

        if cv.waitKey(1) & 0xFF == 27:
            break

if __name__ == "__main__":
    # task1()
    task2()
    camera.release()