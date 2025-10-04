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

        low_color_blue = np.array((100, 90, 90))
        high_color_blue = np.array((140, 220, 250))

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

        low_color_blue = np.array((100, 90, 90))
        high_color_blue = np.array((140, 220, 250))

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

        low_color_blue = np.array((100, 90, 90))
        high_color_blue = np.array((140, 220, 250))

        treshold = cv.inRange(HSV, low_color_blue, high_color_blue)
        erosia  = cv.erode(treshold, core)
        all_morfologic = cv.dilate(erosia, core)
        cv.imshow("Default", img)
        cv.imshow("All", all_morfologic)

        moments = cv.moments(treshold)
        S = moments['m00']
        M01 = moments['m01']
        M10 = moments['m10']
        center_weighted_X = int(M10 / S) if S != 0 else 0
        center_weighted_Y = int(M01 / S) if S != 0 else 0
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

def task5():
    core = np.ones((3, 3), np.uint8)

    while camera.isOpened():
        ok, img = camera.read()
        if not(ok):
            break

        HSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        low_color_blue = np.array((100, 90, 90))
        high_color_blue = np.array((140, 220, 250))

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

        white_pixels = np.argwhere(all_morfologic == 255)

        if white_pixels.size > 0:
            min_x, min_y = white_pixels.min(axis=0)
            max_x, max_y = white_pixels.max(axis=0)

            cv.rectangle(img, (min_y, min_x), (max_y, max_x), (0, 255, 0), 2)

            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2

            cv.circle(img, (center_y, center_x), 4, (0, 0, 255), -1)

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
    # task4()
    task5()
    camera.release()