# Importing packages
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

# Start time for measurements
start = time.time()

# Read input image
image = picam2.capture_array()

# Define image dimensions
width = image.shape[1]
height = image.shape[0]

# Define scale factor for image normalization
scale = 0.00392

# Define width scale for object size adjustment
width_scale = 0.9

# Known width and focal length for distance estimation
KNOWN_WIDTH = 7
focalLength = 820

# Define velocity and angular velocity
v = 140 / 5
w = 360 * 2 / 15
w_pix = 1070

# Read class names from a text file
classes = None
with open("./wer.txt", 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Generate different colors for different classes
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Read pre-trained model and config file
net = cv2.dnn.readNet("./wer.weights", ".wer.cfg")

def get_output_layers(net):
    # Get the output layer names of the network
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    # Draw bounding box on the detected object with class name
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def identify_image(image):
    # Perform object detection on the input image
    image = image.copy()

    # Create input blob
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    # Set input blob for the network
    net.setInput(blob)

    # Run inference through the network and gather predictions from output layers
    outs = net.forward(get_output_layers(net))

    # Initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # Iterate over each detection from each output layer
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w / 2
				y = center_y - h / 2
				class_ids.append(class_id)
				confidences.append(float(confidence))
				boxes.append([x, y, w, h])
			
	# apply non-max suppression
	indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

	# go through the detections remaining
	# after nms and draw bounding box
	for i in indices:
		i = i[0]
		box = boxes[i]
		x = box[0]
		y = box[1]
		w = box[2]
		h = box[3]
		
		draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
	
	return boxes,class_ids,image


def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth

found=False
measured=False
scanning=False
turning=False
angle=0
while True:
	
	if not found and not scanning:
		print("Looking for the object...")
		image=picam2.capture_array()
		boxes,class_ids,mod_image=identify_image(image)
		for i,class_id in enumerate(class_ids):
			if class_id in [40,41,75]:
				target_x=boxes[i][0]
				target_w=boxes[i][2]
				print("object",class_id,"identified at x:",target_x)
				found=True
				cv2.imwrite("object-detection.jpg", mod_image)
				start=time.time()
				a_pix_target=np.abs(image.shape[1]/2-target_x)
				if target_x>image.shape[1]/2:
					fc.turn_right(10)
				else:
					fc.turn_left(10)
				turning=True
				break
		if not found:
			print("object not found")
			cv2.imwrite("object-detection.jpg", image)
			start=time.time()
			scanning=True
			if angle>=2:
				fc.turn_left(10)
			else:
				fc.turn_right(10)
	
	if scanning:
		a=(time.time()-start)*w
		if a>45:
			fc.stop()
			time.sleep(0.5)
			angle+=1
			angle=angle%4
			scanning=False
	
	if turning:
		a_pix=(time.time()-start)*w_pix
		if a_pix>a_pix_target:
			cm=distance_to_camera(KNOWN_WIDTH,focalLength,target_w*width_scale)
			print("travelling",cm,"cm")
			fc.forward(20)
			start=time.time()
			turning=False
	
	if found and not turning:
		d=(time.time()-start)*v
		#print("travelled",d,"cm")
		if d>(cm):
			fc.stop()
			break
		
		
			
				

# wait until any key is pressed
cv2.waitKey()


# release resources
cv2.destroyAllWindows()
