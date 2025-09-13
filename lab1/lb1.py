import cv2 as cv
print(cv.__version__)

#TASK 1
# tuta_image = cv.imread("input/tytuta.jpg")
# cv.namedWindow('Display window', cv.WINDOW_FREERATIO)
# cv.imshow('Display window', tuta_image)
# cv.waitKey(0)
# cv.destroyAllWindows()

#TASK 2
# tuta_image_pixeled = cv.imread("input/tytuta.jpg", cv.IMREAD_REDUCED_COLOR_8)
# tuta_image_negative = cv.imread("input/tytuta.bmp", cv.IMREAD_COLOR_RGB)
# tuta_image_wb = cv.imread("input/tytuta.png", cv.IMREAD_GRAYSCALE)
# cv.namedWindow('free window ratio', cv.WINDOW_FREERATIO)
# cv.namedWindow('normal window', cv.WINDOW_NORMAL)
# cv.namedWindow('autosize window', cv.WINDOW_AUTOSIZE)
# cv.imshow('free window ratio', tuta_image_pixeled)
# cv.imshow('autosize window', tuta_image_negative)
# cv.imshow('normal window', tuta_image_wb)
# cv.waitKey(0)
# cv.destroyAllWindows()

#TASK 3
import time
cap = cv.VideoCapture(r'C:/Users/twink/Desktop/vse/vegas/bannynuke.mp4', cv.CAP_ANY)
cv.namedWindow('frame', cv.WINDOW_FREERATIO)

fps = cap.get(cv.CAP_PROP_FPS)
ms_for_1_fps = int(1000/(fps*1.5))
while(cap.isOpened()):
    start_time_frame = time.perf_counter()
    ret, frame = cap.read()
    if not ret:
        break
    processing_time = time.perf_counter() - start_time_frame
    delay = ms_for_1_fps - int(processing_time * 1000)
    key = cv.waitKey(delay if delay > 1 else 1) & 0xFF
    # frame_disp = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # frame_disp = cv.cvtColor(frame, cv.COLOR_BGR2YUV)
    # frame_disp = cv.cvtColor(frame, cv.COLOR_RGB2XYZ)
    frame_disp = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    frame_re = cv.resize(frame_disp, (1000, 700))
    cv.imshow('frame', frame_re)
    if key & 0xFF == 27:
        break


