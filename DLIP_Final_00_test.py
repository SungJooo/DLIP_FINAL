import torch
import cv2
import random
from PIL import Image
import numpy as np

# Load the Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.classes = [0]

# cap = cv2.VideoCapture('../data/video4.mp4')
cap = cv2.VideoCapture(0)   # Webcam
# cap = cv2.VideoCapture(1)
# cap = cv2.VideoCapture('../data/logi1.mp4')    # logi1, logi2
# cap = cv2.VideoCapture('../data/video0608_l1.MOV') # iphonex1_1, iphonex1_2, iphonex05_1, iphonex05_2


while(1):
    _, frame = cap.read()
    
    results = model(frame) 

    cv2.imshow("result", frame)
    cv2.imshow("result", np.squeeze(results.render()))


    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break