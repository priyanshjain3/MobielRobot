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
camera = Picamera2()
camera.preview_configuration.main.size = (1280, 720)
camera.preview_configuration.main.format = "RGB888"
camera.preview_configuration.align()
camera.configure("preview")

# Start the camera preview
camera.start()

# Stop the car initially
fc.stop()

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
    dist = (knownWidth * focalLength) / perWidth
    return dist

# Initialize the known distance from the camera to the object
KNOWN_DISTANCE = 100
# Initialize the known object width
KNOWN_WIDTH = 7

# Capture the first image
image = camera.capture_array()
# Find the marker in the image and initialize the focal length
marker = find_marker(image)
focalLength = 820

v = 140 / 5

power_val = 50
key = 'status'

def readchar():
    # Read a single character from the user input
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def readkey(getchar_fn=None):
    # Read a key from the user input
    getchar = getchar_fn or readchar
    c1 = getchar()
    if ord(c1) != 0x1b:
        return c1
    c2 = getchar()
    if ord(c2) != 0x5b:
        return c1
    c3 = getchar()
    return chr(0x10 + ord(c3) - 65)

measured = False
while True:
    if not measured:
        d = []
        for i in range(20):
            image = camera.capture_array()
            marker = find_marker(image)
            d.append(distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0]))
        cm = np.median(d)
        
        box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
        box = np.int0(box)
        cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
        cv2.putText(image, "%.0fcm" % (cm),(image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIM
		cv2.imshow("image", image)
		cv2.imwrite("object-detection.jpg", image)
		print(cm)
		if cm<200 and np.std(d)<10:
			print("Travelling",cm,"cm")
			measured=True
			
			fc.forward(20)
			start=time.time()
	if measured:
		d=(time.time()-start)*v
		#print("travelled",d,"cm")
		if d>(cm-10):
			fc.stop()
			break
	
	if cv2.waitKey(1) == ord("q"):
		break
cv2.destroyAllWindows()
