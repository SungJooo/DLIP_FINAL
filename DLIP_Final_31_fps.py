import torch
import cv2
import random
from PIL import Image
import numpy as np
import time

# Load the Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.classes = [0]

# cap = cv2.VideoCapture('../data/video4.mp4')
cap = cv2.VideoCapture(0)   # Webcam
# cap = cv2.VideoCapture(1)
# cap = cv2.VideoCapture('../data/logi1.mp4')    # logi1, logi2
# cap = cv2.VideoCapture('../data/video0608_l1.MOV') # iphonex1_1, iphonex1_2, iphonex05_1, iphonex05_2

fpsStart = 0
fps = 0

while(1):
    ret, frame = cap.read()
    
    if ret:
        results = model(frame) 
    else:
        break

    fpsEnd = time.time()
    timeDiff = fpsEnd - fpsStart
    fps = 1/timeDiff
    fpsStart = fpsEnd

    fpsText  = "FPS: {:2.0f}".format(fps)
    cv2.putText(frame, fpsText, (30, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("result", frame)
    cv2.imshow("result", np.squeeze(results.render()))


    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break