import os
from datetime import datetime
from gtts import gTTS
import cv2
import face_recognition
import numpy as np

path = 'IA'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def giveInfo(name):
    with open('INFO.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.write(f'\n{name},{dtString}')
            myTexr = (name)
            language = 'en'
            output = gTTS(text=myTexr, lang=language, slow=False)
            output.save("output.mp3")
            os.system("start output.mp3")






encodeListKnown = findEncodings(images)
print('encoding complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    facesCurrntFrame = face_recognition.face_locations(imgS)
    encodeCurrntFrame = face_recognition.face_encodings(imgS,facesCurrntFrame)

    for encodeFace,faceLoc in zip(encodeCurrntFrame,facesCurrntFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()

            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1+0,x2+25,y2+50,x1+0
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(255,0,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-2),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            giveInfo(name)
            # myTexr = name
            # language = 'en'
            # output = gTTS(text=myTexr, lang=language, slow=False)
            # output.save("output.mp3")
            
            # os.system("start output.mp3")


    cv2.imshow('Webcam',img)
    # cv2.waitKey(1)
    if cv2.waitKey(10) & 0xFF == ord('q'):
            break







