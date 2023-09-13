import cv2 as cv
import face_recognition
import pickle
import os

face = r"C:\Users\akhsh\Desktop\Project\Fun\Faces"
humans = []
featured = []

for filename in os.listdir(face):
    print(os.path.join(face,filename))
    image = cv.imread(os.path.join(face,filename))
    image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
    name = os.path.splitext(filename)[0]
    humans.append(image)
    feature = face_recognition.face_encodings(image)[0]
    featured.append([feature,name])
    print("Faces added: ",len(featured))


file = open("faces.p","wb")
pickle.dump(featured,file)
file.close()


