from detection import detection
import cv2
import json
import time

detector = detection()

video_path = "data/elephant.mp4"
video = cv2.VideoCapture(video_path)

while video.isOpened():

    ret, frame = video.read()

    if ret:

        boxes, scores, classe = detector.predict(frame)
        detector.visual(frame, boxes, classe, scores)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video.release()
cv2.destroyAllWindows()