import cv2
import face_recognition as fr
import numpy as np
import os
from datetime import datetime


path = r'C:\Users\Pilotech\PycharmProjects\opencvv\attandence'

images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    img = cv2.imread(f'{path}/{cl}')
    images.append(img)
    classNames.append(os.path.splitext(cl)[0])


def findEncod(Images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
encodeListKnown = findEncod(images)

def Attendance(name):
    with open('Attendence.csv','r+') as f:
        datalist = f.readlines()
        namelist = []
        for line in datalist:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            datatime = now.strftime('%H:%M:%d:%b')
            f.writelines(f'\n{name},{datatime},P')

vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    _,img = vid.read()
    imgresize = cv2.resize(img,(0,0),None,0.3,0.3)
    imgresize = cv2.cvtColor(imgresize, cv2.COLOR_BGR2RGB)
    faceframe = fr.face_locations(imgresize)
    encodeframe = fr.face_encodings(imgresize,faceframe)

    for encodeFace, Faceloc in zip(encodeframe,faceframe):
        match = fr.compare_faces(encodeListKnown,encodeFace)
        faceDis = fr.face_distance(encodeListKnown,encodeFace)
        matchIndex = np.argmin(faceDis)

        if match[matchIndex]:
            name = classNames[matchIndex].upper()
            y1,x2,y2,x1 = Faceloc
            y1, x2, y2, x1 = y1*3, x2*4, y2*4, x1*3
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.rectangle(img,(x1,y2-40),(x2,y2),(0,0,255),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),2)
            Attendance(name)
    cv2.imshow('AI & ML Attendance',img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break