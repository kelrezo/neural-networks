import numpy as np
import argparse
import cv2



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())



image = cv2.imread(args["image"])


boundaries = [
	([0, 10, 150], [0, 100, 255]),
	([86, 31, 4], [220, 88, 50]),
	([25, 146, 190], [62, 174, 250]),
	([103, 86, 65], [145, 133, 128])
]
#boundaries = [([0, 204,204], [51, 255, 255])]
boundaries = [([14, 126, 180], [34, 146, 255])]
lower = np.array(boundaries[0][0], dtype = "uint8")
upper = np.array(boundaries[0][1], dtype = "uint8")


cap = cv2.VideoCapture(0)

while True:
	ret, frame = cap.read()
# create NumPy arrays from the boundaries
	#lower = np.array(lower, dtype = "uint8")
	#upper = np.array(upper, dtype = "uint8")

	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv2.inRange(frame, lower, upper)
	output = cv2.bitwise_and(frame, frame, mask = mask)
	#cv2.imshow('video',output)
	# show the images
	cv2.imshow("images", np.hstack([frame, output]))
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()





'''
# loop over the boundaries
for (lower, upper) in boundaries:
	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")

	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv2.inRange(image, lower, upper)
	output = cv2.bitwise_and(image, image, mask = mask)

	# show the images
	cv2.imshow("images", np.hstack([image, output]))
	cv2.waitKey(0)

'''