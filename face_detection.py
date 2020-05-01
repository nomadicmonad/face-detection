import cv2
import sys
import urllib.request
import os
import time 

save_location = "/detected_faces"
xml_file_path = "haarcascade_frontalface_default.xml"
while (not os.path.exists(xml_file_path)):
    time.sleep(1)
faceCascade = cv2.CascadeClassifier(xml_file_path)
video_capture = cv2.VideoCapture(0)

# Basic code adapted from: https://github.com/MertArduino/RaspberryPi-Mertracking/blob/master/mertracking.py
count = 0
while True:
    # Capture the video
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = []
    try:
        faces = faceCascade.detectMultiScale(gray, 1.3, 2) 
    except Exception as e:
        print(e)

    # Draw a rectangle around the faces
    detected = False
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        detected = True
        count += 1
    image_save_path = os.path.join(save_location,"detected_{}.png".format(count%1000))
    cv2.imwrite(image_save_path,frame)
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release Capture
video_capture.release()
cv2.destroyAllWindows()