import cv2


video = cv2.VideoCapture("big-object.mp4")
tracker = cv2.TrackerKCF_create()

success, frame = video.read()

frame = cv2.resize(frame, (800, 600))
bbox = cv2.selectROI("ROI", frame, False)

tracker.init(frame, bbox)

while True:
    success, frame = video.read()

    if not success:
        break

    frame = cv2.resize(frame, (800, 600))
    success, bbox = tracker.update(frame)

    if success:
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    cv2.imshow("KCF Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()