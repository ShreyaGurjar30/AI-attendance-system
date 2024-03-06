import face_recognition
import cv2
import numpy as np
import csv
import os 

from datetime import datetime
import pandas as pd

video_capture = cv2.VideoCapture(1) #to take input from camera

william_image= face_recognition.load_image_file("Face_recognise_NOGUI/photos/willaim.jpg")
william_encoding = face_recognition.face_encodings(william_image)[0]

jeff_image= face_recognition.load_image_file("Face_recognise_NOGUI/photos/jeff.jpg")
jeff_encoding = face_recognition.face_encodings(jeff_image)[0]

mayur_image = face_recognition.load_image_file("Face_recognise_NOGUI/photos/mayur.jpg")
mayur_encoding = face_recognition.face_encodings(mayur_image) [0]

narendra_image = face_recognition.load_image_file("Face_recognise_NOGUI/photos/narendra.jpg")
narendra_encoding = face_recognition.face_encodings(narendra_image) [0]


known_face_encoding = [
william_encoding,
jeff_encoding,
mayur_encoding,
narendra_encoding
]

known_faces_names = [
"willaim Smith",
"Jeff Bezos",
"Mayur Gaikwad",
"Narendra Modi"
]

students = known_faces_names.copy()
face_locations = []
face_encodings = []
face_names = []
s=True

now = datetime.now()
current_date = now.strftime("%#x")

f = open(current_date+'.csv','w+',newline='')
lnwriter = csv.writer(f)

while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s :
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index] :
                name = known_faces_names[best_match_index]
                
            face_names.append(name)
            if name in known_faces_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    # print(name)
                    current_time = now.strftime("%#c")
                    lnwriter.writerow((name,current_time))
                    # print(current_time)
                    # present_stud = {
                #         "Name" : name,
                #         "Date" : current_date
                #     }
                #     # print(present_stud)
                #     detail_list = []
                #     detail_list.append(present_stud)
                #     df= pd.DataFrame(detail_list)
                #     df.to_excel('Presentee.xlsx')
                # detail_list.clear()        
    cv2.imshow("attendence system",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): #q is the key to quit the window
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
# cv2.raise_exception() 
# cv2.join()