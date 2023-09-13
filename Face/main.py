import face_recognition
import cv2 as cv
import pickle
import numpy as np

unkk = [[],[],[]]

file = open(r"C:\Users\akhsh\Desktop\Project\Fun\faces.p","rb")
items = pickle.load(file)
faces , id = items
g=0

def appear(unk,check,faces,face_pos,id):

    enc_face = face_recognition.face_encodings(check,face_pos)
    for enc,loc in zip(enc_face,face_pos):
        match = face_recognition.compare_faces(faces,enc)

        if any(match)==0:
            for enc,loc in zip(enc_face,face_pos):
                match = face_recognition.compare_faces(unk[0],enc)
            
            print("already : " , match)
            if any(match)==0:
                print("New Unkown")
                unk[0].append(enc)
                unk[1].append(id)
                unk[2].append(1)
                return 1
            
            else:
                print("Ive seen him")
                s = np.argmax(np.array(match))
                unk[2][s]+=1
            

cap = cv.VideoCapture(r"C:\Users\akhsh\Desktop\Project\Fun\Funny\Streamlit\videodata\p1p8.mp4")
curr_frame = 0
while 1:
    ret , frame = cap.read()
    if not ret:
        break
    if curr_frame % 8 ==0:
        face_pos = face_recognition.face_locations(frame,model="hog")
        if len(face_pos)!=0:
            unk_unk = appear(unkk,frame,faces,face_pos,g)
            if unk_unk:
                g+=1

        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    if curr_frame % 1000 ==0:
        curr_frame=1
        print(unkk)

    curr_frame += 1
cap.release()
cv.destroyAllWindows()


