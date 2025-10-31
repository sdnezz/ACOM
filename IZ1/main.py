import cv2
import sys
import os

def object_tracking(method_name: str, video_name: str, video_dimension: tuple):
    tracker = None

    if method_name == "CSRT":
        tracker = cv2.legacy.TrackerCSRT.create()
    elif method_name == "KCF":
        tracker = cv2.legacy.TrackerKCF.create()

    video = cv2.VideoCapture(f"{video_name}.mp4")
    original_fps = video.get(cv2.CAP_PROP_FPS)
    print(f"FPS исходного видео: {original_fps}")


    success, frame = video.read()
    frame = cv2.resize(frame, video_dimension)
    bbox = cv2.selectROI("ROI", frame, False)
    cv2.destroyWindow("ROI")

    tracker.init(frame, bbox)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'{method_name};{video_name}-tracking.mp4', fourcc, original_fps, video_dimension)

    while video.isOpened():
        success, frame = video.read()

        if not success:
            break

        frame = cv2.resize(frame, video_dimension)
        success, bbox = tracker.update(frame)

        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        out.write(frame)

        cv2.imshow('Tracking', frame)

        char = cv2.waitKey(5)
        if char == 27:
            break


    video.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # object_tracking("KCF", "big-object", (1920, 1080))
    object_tracking("CSRT", "big-object", (1920, 1080))