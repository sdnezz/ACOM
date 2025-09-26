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

        low_color_blue = np.array((100, 90, 100))
        high_color_blue = np.array((140, 190, 190))

        treshold = cv.inRange(HSV, low_color_blue, high_color_blue)

        cv.imshow("Default", img)
        cv.imshow("only BLUE", treshold)

        if cv.waitKey(1) & 0xFF == 27:
            break

def task3():
    core = np.ones((3, 3), np.uint8)
    while camera.isOpened():
        ok, img = camera.read()
        if not(ok):
            break

        HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        low_color_blue = np.array((100, 90, 100))
        high_color_blue = np.array((140, 190, 190))

        treshold = cv.inRange(HSV, low_color_blue, high_color_blue)

        cv.imshow("Default", img)
        cv.imshow("BLUE", treshold)

        erosia  = cv.erode(treshold, core)
        dilatation = cv.dilate(treshold, core)

        cv.imshow("ERODE", erosia)
        cv.imshow("DILATE", dilatation)

        all_morfologic = cv.dilate(erosia, core)

        cv.imshow("All", all_morfologic)
        if cv.waitKey(1) & 0xFF == 27:
            break

def task4():
    core = np.ones((3, 3), np.uint8)

    while camera.isOpened():
        ok, img = camera.read()
        if not(ok):
            break

        HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        low_color_blue = np.array((100, 90, 100))
        high_color_blue = np.array((140, 190, 190))

        treshold = cv.inRange(HSV, low_color_blue, high_color_blue)
        erosia  = cv.erode(treshold, core)
        all_morfologic = cv.dilate(erosia, core)
        cv.imshow("All", all_morfologic)

        moments = cv.moments(treshold)
        S = moments['m00']
        M01 = moments['m01']
        M10 = moments['m10']
        center_weighted_X = int(M10 / S) if S != 0 else 0
        center_weighted_Y = int(M01 / S) if S != 0 else 0
        cv.circle(img, center=(center_weighted_X, center_weighted_Y), radius=4, color=(0, 255, 0), thickness=-1)

        contours, _ = cv.findContours(all_morfologic, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        if contours:
            largest_contour = max(contours, key=cv.contourArea)

            x, y, w, h = cv.boundingRect(largest_contour)

            cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)

        cv.imshow("Default", img)
        info_display = np.zeros((200, 400, 3), dtype=np.uint8)
        cv.putText(info_display, f"S: {S}", (10, 30),
                    cv.FONT_HERSHEY_COMPLEX, 0.7, (120, 0, 0), 2)
        cv.putText(info_display, f"M01 (X * intensity): {M01}", (10, 70),
                    cv.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
        cv.putText(info_display, f"M10 (Y * intensity): {M10}", (10, 110),
                    cv.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
        cv.putText(info_display, f"Center Weighted: ({center_weighted_X}, {center_weighted_Y})", (10, 150),
                    cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
        cv.imshow("Info", info_display)

        if cv.waitKey(1) & 0xFF == 27:
            break

if __name__ == "__main__":
    # task1()
    # task2()
    # task3()
    task4()
    camera.release()