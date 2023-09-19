import FacialRecognition as face
import numpy as np
import cv2

video_capture = cv2.VideoCapture("sample.mp4")

while True:
    ret, frame = video_capture.read() #ret is success or not = true,false 
    if ret:
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break