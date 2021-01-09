# -*- coding: utf-8 -*-
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def detect(gray, frame):
    # get coordinates of the rectangle that will detect the face 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # draw the rectangles
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # detect the eyes 
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            # draw the rectangles
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (255,0,0), 2)
            
    return frame

# face recognition with webcam
video_capture = cv2.VideoCapture(0)

while video_capture.isOpened():
    _, frame = video_capture.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows()
    
    