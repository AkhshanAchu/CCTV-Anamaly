from ultralytics import YOLO
import cv2 as cv
import pickle
import numpy as np

model = YOLO(r"C:\Users\akhsh\Desktop\Project\Fun\Funny\Streamlit\best.pt")


unk = {0:0,2:0}


def appear(unk,classes):
    if any(classes):
        for i in classes:
            if i==0:
                unk[0]+=1
            if i==2:
                unk[2]+=1



cap = cv.VideoCapture(r"C:\Users\akhsh\Downloads\videoplayback (1).mp4")
curr_frame = 0
while 1:
    ret , frame = cap.read()
    if not ret:
        break
    if curr_frame % 8 ==0:
        result = model.predict(frame)
        for i in result:
            boxes = [j.xyxy[0] for j in i.boxes]
            clss = [j.cls for j in i.boxes]
        unk_unk = appear(unk,clss)

    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    if unk[0]>10:
        print("Knife")
        unk[0]=0

    if unk[2]>10:
        unk[2]=0
        print("Gun")

    if curr_frame % 100==0:
        curr_frame=0
        unk[0]=0
        unk[2]=0

    curr_frame += 1
cap.release()
cv.destroyAllWindows()




