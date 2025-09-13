import cv2 as cv
print(cv.__version__)

#TASK 1
def CVImage():
    tuta_image = cv.imread("input/tytuta.jpg")
    cv.namedWindow('Display window', cv.WINDOW_FREERATIO)
    cv.imshow('Display window', tuta_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

#TASK 2
def CVImageForm():
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

#TASK 3
import time
def CVideCapture():
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

#TASK 4
def readIPWriteTOFile():
    video = cv.VideoCapture(r'C:/Users/twink/Desktop/vse/vegas/bannynuke.mp4', cv.CAP_ANY)
    if not video.isOpened():
        return
    ok, img = video.read()
    w = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video_writer = cv.VideoWriter('output/output.mp4', fourcc, 25, (w, h))
    cv.namedWindow("img", cv.WINDOW_FREERATIO)
    while ok:
        ok, img = video.read()
        cv.imshow("img", img)
        video_writer.write(img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv.destroyAllWindows()

#TASK 5
def CV_IMAGE_COMPARE_HSV():
    frame = cv.imread("input/tytuta.jpg")
    cv.namedWindow('HSV', cv.WINDOW_FREERATIO)
    cv.namedWindow('BGR', cv.WINDOW_FREERATIO)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    cv.imshow('HSV', hsv)
    cv.imshow('BGR', frame)
    cv.waitKey(0)
    cv.destroyAllWindows()

#TASK 7
def CV_WEBCAM():
    camera = cv.VideoCapture(0, cv.CAP_DSHOW)
    if not camera.isOpened():
        return
    ok, img = camera.read()
    weight = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    video_writer = cv.VideoWriter('output/output_camera.mp4', fourcc, 25, (weight, height))
    cv.namedWindow("img", cv.WINDOW_FREERATIO)

    centerX = weight // 2
    centerY = height // 2
    vertical_w, vertical_h = max(40, weight//12), height//3
    horizont_w, horizont_h = weight//2, max(40, height//12)
    color, thickness = (0,0,255), 1

    while ok:
        ok, img = camera.read()
        # cv.line(img, (centerX - 20, centerY), (centerX + 20, centerY), (0, 0, 255), 2)
        cv.rectangle(img,
                     (centerX - vertical_w//2, centerY - vertical_h//2),
                     (centerX + vertical_w//2, centerY + vertical_h//2),
                     color, thickness=thickness, lineType=cv.LINE_AA)

        cv.rectangle(img,
                     (centerX - horizont_w//2, centerY - horizont_h//2),
                     (centerX + horizont_w//2, centerY + horizont_h//2),
                     color, thickness=thickness, lineType=cv.LINE_AA)

        cv.imshow("img", img)
        video_writer.write(img)
        if cv.waitKey(1) & 0xFF == 27:
            break

    camera.release()
    video_writer.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    # CVImage()
    # CVImageForm()
    # CVideCapture()
    # readIPWriteTOFile()
    # CV_IMAGE_COMPARE_HSV()
    CV_WEBCAM()