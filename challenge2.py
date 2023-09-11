# import the necessary packages
from imutils import paths
import numpy as np
import imutils
import cv2

import picar_4wd as fc
import sys
import tty
import termios
import asyncio
import time

from picamera2 import Picamera2

# Initialize the PiCamera
picam2 = Picamera2()

# Configure camera preview settings
picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

def find_marker(image):
    # Convert the image to grayscale, blur it, and detect edges
    gauss = image.copy()
    gauss = cv2.GaussianBlur(gauss, (5, 5), 0)
    edges = cv2.Canny(gauss, 30, 125)
    
    # Find the contours in the edged image and keep the largest one
    cnts = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    
    # Compute the bounding box of the paper region and return it
    return cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
    # Compute and return the distance from the marker to the camera
    return (knownWidth * focalLength) / perWidth

# Initialize the known distance from the camera to the object
KNOWN_DISTANCE = 100
# Initialize the known object width
KNOWN_WIDTH = 7

# Load the first image that contains an object at a known distance from the camera
# Find the paper marker in the image and initialize the focal length
image = picam2.capture_array()
marker = find_marker(image)
focalLength = 820  # marker[1][0] * KNOWN_DISTANCE / KNOWN_WIDTH

v = 140 / 5
w = 360 * 2 / 15

power_val = 50
key = 'status'

measured = False
stage = 0
while True:
    if not measured:
        d = []
        for i in range(20):
            image = picam2.capture_array()
            marker = find_marker(image)
            d.append(distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0]))
        
        cm = np.median(d)
        
        box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
        box = np.int0(box)
        cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
        cv2.putText(image, "%.0fcm" % (cm), (image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 3)
        
        cv2.imshow("image", image)
        cv2.imwrite("object-detection.jpg", image)
        print(cm)
        if cm < 200 and np.std(d) < 10:
            print("Travelling", cm, "cm")
            measured = True
            
            fc.forward(20)
            start = time.time()
            
    if measured:
        if stage == 0:
            d = (time.time() - start) * v
            if d > (cm - 20):
                fc.turn_right(10)
				start=time.time()
				stage=1
		if stage==1:
			a=(time.time()-start)*w
			if a>45:
				fc.forward(20)
				start=time.time()
				stage=2
		if stage==2:
			d=(time.time()-start)*v
			if d>30:
				fc.turn_left(20)
				start=time.time()
				stage=3
		if stage==3:
			d=(time.time()-start)*v
			if d>20:
				fc.turn_left(20)
				start=time.time()
				stage+=1
		if stage==4:
			a=(time.time()-start)*w
			if a>45:
				fc.forward(20)
				start=time.time()
				stage+=1
		if stage==5:
			d=(time.time()-start)*v
			if d>30:
				fc.turn_right(20)
				start=time.time()
				stage+=1
		if stage==6:
			a=(time.time()-start)*w
			if a>45:
				fc.forward(20)
				start=time.time()
				stage+=1
		if stage==7:
			d=(time.time()-start)*v
			if d>70:
				fc.stop()
				stage+=1
	if cv2.waitKey(1) == ord("q"):
		break
cv2.destroyAllWindows()
