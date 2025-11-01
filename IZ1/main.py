import cv2
import sys
import os
from camshift import camshift

def object_tracking(method_name: str, video_name: str, video_dimension: tuple):
    tracker = None

    if method_name == "CSRT":
        tracker = cv2.legacy.TrackerCSRT.create()
    elif method_name == "KCF":
        tracker = cv2.legacy.TrackerKCF.create()

    video = cv2.VideoCapture(f"{video_name}.mp4")
    original_fps = video.get(cv2.CAP_PROP_FPS)

    success, frame = video.read()
    frame = cv2.resize(frame, video_dimension)
    bbox = cv2.selectROI("ROI", frame, False)
    cv2.destroyWindow("ROI")

    tracker.init(frame, bbox)

    output_dir = method_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"{video_name}-tracking.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, original_fps, video_dimension)

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

        # cv2.imshow('Tracking', frame)

        char = cv2.waitKey(5)
        if char == 27:
            break


    video.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    object_tracking("KCF", "resizable", (800, 600))
    object_tracking("CSRT", "resizable", (800, 600))
    camshift("resizable", (800, 600))

    object_tracking("KCF", "fast", (800, 600))
    object_tracking("CSRT", "fast", (800, 600))
    camshift("fast", (800, 600))

    object_tracking("KCF", "escape_object", (800, 600))
    object_tracking("CSRT", "escape_object", (800, 600))
    camshift("escape_object", (800, 600))

    object_tracking("KCF", "many_objects", (800, 600))
    object_tracking("CSRT", "many_objects", (800, 600))
    camshift("many_objects", (800, 600))

    object_tracking("KCF", "small_meteor", (800, 600))
    object_tracking("CSRT", "small_meteor", (800, 600))
    camshift("small_meteor", (800, 600))