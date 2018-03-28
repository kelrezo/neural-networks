import cv2
import numpy as np

green = np.uint8([[[0,255,0 ]]])

hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
x=hsv_green[0][0][0]
y =hsv_green[0][0][1]
z = hsv_green[0][0][2]
lower = np.array([x-10,y,z])
upper = np.array([x+10,y,z])

print(lower,upper)

x = np.array([1,2,3,4,5,6,7,8,9])
print(len(x) is not None)