import numpy as np
import cv2, os
import face_recognition


path = r'attendance'
images = []
classNames = []
mylist = os.listdir(path)

for cl in mylist:
    curimg = cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print('encode complete')
print(len(encodeListKnown))

cap = cv2.VideoCapture('rtsp://admin:ZECTUM@10.1.1.60:554/H.264')
# cap = cv2.VideoCapture(0)

while True:
    try:
        sucess, img = cap.read()
        imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            facDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            # print(facDis)
            matchIndex = np.argmin(facDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                # print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 1)
                cv2.rectangle(img, (x1, y2-35), (x2,y2), (0,255,0), cv2.FILLED)
                cv2.putText(img,name,(x1+6, y2-6),  cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255), 1)
        ims = cv2.resize(img,  (960,540))
        cv2.imshow('Camera', ims)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        continue
cap.release()
cv2.destroyAllWindows()
