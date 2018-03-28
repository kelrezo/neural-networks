import cv2
import sys
import numpy as np
from webcam import *


'''
filename = 'chessboard.png'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
dst = cv2.dilate(dst,None)
img[dst>0.01*dst.max()]=[0,0,255]
cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
	cv2.destroyAllWindows()

'''



face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while True:
	ret, frame = cap.read()
	#img = cv2.imread(frame)

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#print(frame)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	print(faces)
	for (x,y,w,h) in faces:
	    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
	    roi_gray = gray[y:y+h, x:x+w]
	    roi_color = frame[y:y+h, x:x+w]
	    eyes = eye_cascade.detectMultiScale(roi_gray)
	    for (ex,ey,ew,eh) in eyes:
	        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	cv2.imshow('video',frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
