import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

path = 'Image attendance'
images = []
class_names = {}
my_list = os.listdir(path)
print("Total Classes Detected:", len(my_list))
print("Classes in the directory:")
for item in my_list:
    print(item)
for i, item in enumerate(my_list):
    cur_img = cv2.imread(f'{path}/{item}')
    images.append(cur_img)
    class_name = os.path.splitext(item)[0]
    class_names[class_name] = face_recognition.face_encodings(cur_img)[0]

print('Encodings Complete')

present_list = []

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(list(class_names.values()), face_encoding)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = list(class_names.keys())[first_match_index]
            if name not in present_list:
                present_list.append(name)

        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(img, name, (left, top - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in  nameList:
            now = datetime.now()
            dt_string = now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{dt_string}')

for name in present_list:
    markAttendance('Parth')

cap.release()
cv2.destroyAllWindows()
