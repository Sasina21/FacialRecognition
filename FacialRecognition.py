import face_recognition as face
import numpy as np
import cv2

video_capture = cv2.VideoCapture("sample.mp4")

pop_image = face.load_image_file("pop.jpg")
pop_face_encoding = face.face_encodings(pop_image)[0]

face_location = []
face_encodings = []
face_names = []
face_percenrt = []
process_this_frame = True

known_face_encodings = [pop_face_encoding]
known_face_names = ["PONGKUL"]

while True:
    ret, frame = video_capture.read() #ret is success or not = true,false 
    if ret:
        small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        rgb_small_frame = small_frame[:,:,::-1]

        face_name = []
        face_percent = []

        if process_this_frame:
            face_location = face.face_locations(rgb_small_frame, model="cnn")
            face_endcodings = face.face_encodings(rgb_small_frame, face_location)

            for face_encoding in face_encodings:
                face_distance = face.face_distance(known_face_encodings, face_encoding)
                best = np.argmin(face_distance)
                face_percent_value = 1 - face_distance[best]

                if face_percent_value >= 0.5:
                    name = known_face_names[best]
                    percent = round(percent*100,2)
                    face_percent.append(percent)
                
                else:
                    name = "UNKNOWN"
                    face_percent.append(0)
                face_name.append(name)
        
        for(top, right, bottom, left), name, percent in zip(face_location, face_names, face_percent):
            top = 2
            right = 2
            bottom = 2
            left = 2

            if name == "UNKNOWN":
                color = [45,2,209]
            else:
                color = [255,102,51]

            cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
            cv2.rectangle(frame, (left-1, top -30), (right-1, top), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left-6, top-6), font, 0.6, (255, 255,255), 1)
            cv2.putText(frame, "MATCH: "+ str(percent) + "%", (left-6, bottom+23), font, 0.6, (255, 255,255), 1)


        process_this_frame = not process_this_frame
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break