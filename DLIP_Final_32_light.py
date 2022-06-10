import torch
import cv2
import random
from PIL import Image
import numpy as np
import math
import time

# Load the Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
# model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
model.classes=[0]
# cap = cv2.VideoCapture('../data/logi2.mp4')
# cap = cv2.VideoCapture('../data/video0608_l1.MOV')   # video0608_l1, video0608_h1, video0608_h2

cap = cv2.VideoCapture(0)
frameno=0
num_people=0
fpsStart = 0
fps = 0



# returns coordinates of box as list
def box_coords(box):
    xmin=int(box[0])
    ymin=int(box[1])
    xmax=int(box[2])
    ymax=int(box[3])
    return [xmin, ymin, xmax, ymax]

# checks if box touches the bottom of frame
def checkbot_box(coords,height):
    ymax=coords[3]
    if ymax>height-(height/54):
        return 1
    else:
        return 0

# returns center coordinates of box
def box_cent(coords):
    cent_x=int((coords[0]+coords[2])/2)
    cent_y=int((coords[1]+coords[3])/2)
    return [cent_x,cent_y]

# gets intersecting area of two boxes
def inters_area(coord1,coord2):
    xmin1=coord1[0]
    ymin1=coord1[1]
    xmax1=coord1[2]
    ymax1=coord1[3]
    xmin2=coord2[0]
    ymin2=coord2[1]
    xmax2=coord2[2]
    ymax2=coord2[3]
    dx=min(xmax1,xmax2)-max(xmin1,xmin2)
    dy=min(ymax1,ymax2)-max(ymin1,ymin2)
    if (dx>0) and (dy>0):
        return dx*dy
    else:
        return 0

# returns list of coordinates of boxes in current frame that are new (no corresponding box in previous frame)
def newbox(coordlist,i_list):
    new_list=[]
    for k in coordlist:
        if k not in [i[0] for i in i_list]:
            new_list+=[k]
    return new_list

# returns list of coordinates of boxes in previous frame that have disappeared (no corresponding box in current frame)
def dispbox(prev_coordlist,i_list):
    disp_list=[]
    for k in prev_coordlist:
        if k not in [i[1] for i in i_list]:
            disp_list+=[k]
    return disp_list

# finds which box in previous slide is the one in current frame (highest intersecting area)
def matchboxes(coordlist,prev_coordlist,width):
    i_list=[]
    for coord in coordlist:
        area=0
        add_ilist=[]
        for prev_coord in prev_coordlist:
            if inters_area(coord,prev_coord)>area and (math.dist(box_cent(coord),box_cent(prev_coord))<(width/20)):
                area=inters_area(coord,prev_coord)
                add_ilist=[[coord, prev_coord]]
            if coord not in [i[0] for i in i_list] and prev_coord not in [j[1] for j in i_list]:
                i_list+=add_ilist
    return i_list


import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
pin_num = 21
GPIO.setup(pin_num, GPIO.OUT, initial=GPIO.LOW)

def GPIO_LIGHT(numPeople, frame):
    if numPeople > 0: GPIO.output(pin_num, GPIO.HIGH)
    else: GPIO.output(pin_num, GPIO.LOW)

    if numPeople > 0: cv2.circle(frame, (int(width*0.9), int(height*0.9)), radius=30, color=(255,255,255), thickness=cv2.FILLED)
    else: cv2.circle(frame, (int(width*0.9), int(height*0.9)), radius=30, color=(0,0,0), thickness=cv2.FILLED)      



    
prev_num_bot=0

while(1):
    frameno+=1
    _, frame = cap.read()

    #frame dimensions
    height=frame.shape[0]
    width=frame.shape[1]
    
    # number of boxes touching bottom
    num_bot=0
    # sendard('0')


    results = model(frame)
    if frameno==1:
        prev_results=results
    
    # # send arduino command
    # if len(results.xyxy[0])>0:
    #     sendard('1')
    # else:
    #     sendard('0')

    # create lists of all box coordinates in previous and current frame
    prev_coordlist=[]
    for j in range(len(prev_results.xyxy[0])):
        prev_coords=box_coords(prev_results.xyxy[0][j])
        prev_coordlist+=[prev_coords]
    coordlist=[]
    for k in range(len(results.xyxy[0])):
        coords=box_coords(results.xyxy[0][k])
        coordlist+=[coords]
    
    for c in coordlist:
        cv2.rectangle(frame,(c[0],c[1]),(c[2],c[3]),(255,0,0),5)
    # list of boxes that have corresponding boxes in previous frame
    i_list=matchboxes(coordlist, prev_coordlist,width)
    # print(i_list)
    # get list of boxes that are new in the frame
    new_list=newbox(coordlist,i_list)
    # print(new_list)
    # print(len(new_list))
    # get list of boxes that have disappeared
    disp_list=dispbox(prev_coordlist,i_list)
    for new_coords in new_list:
        if checkbot_box(new_coords,height)==1:
            num_people-=1
            cv2.rectangle(frame,(new_coords[0],new_coords[1]),(new_coords[2],new_coords[3]),(0,0,255),5)
    
    for disp_coords in disp_list:
        if checkbot_box(disp_coords,height)==1:
            num_people+=1
            cv2.rectangle(frame,(disp_coords[0],disp_coords[1]),(disp_coords[2],disp_coords[3]),(0,255,0),5)
    

    # send rasp GPIO command
    # if num_people>0:
    #     cv2.circle(frame, (int(width*0.9), int(height*0.9)), radius=30, color=(255,255,255), thickness=cv2.FILLED)
    # else:
    #     cv2.circle(frame, (int(width*0.9), int(height*0.9)), radius=30, color=(0,0,0), thickness=cv2.FILLED)    
    GPIO_LIGHT(num_people, frame)
    

    num_peopletxt="Number of people: "+str(num_people)
    cv2.putText(frame, num_peopletxt, (int(width/40), height-int(width/40)), cv2.FONT_HERSHEY_SIMPLEX, round(width/1000), (0, 0, 255), round(width/1000), cv2.LINE_AA)
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.imshow("result", frame)
    # cv2.imshow("result", np.squeeze(results.render()))
    # prev_num_bot=num_bot
    prev_results=results
    # print(type(results.xyxy[0])) #torch.Tensor
    # print(results.xyxy[0][0])
    # print(box_coords(results.xyxy[0][0]))

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        # sendard('0')
        break
    if k == 114 or k == 82:
        num_people = 0