import cv2
import numpy as np
import pyautogui
import ECG_CNN as cnn
from config import *

# cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

# Load Yolo
net = cv2.dnn.readNet(YOLO_MODEL, YOLO_CONFIG)

# Name custom object
classes = ["ecg"]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

risk_colors = [None, (0,0,255), (0,255,255), (255,0,0), ]
risk_text = ['Sano', 'Onda-S', 'Onda-T', 'Onda-Q']

while True:
	# ret,frame = cap.read()
	
	# if ret == False: break


	img =   np.array(pyautogui.screenshot()) 
	img = cv2.resize(img, (1280, 720))	
	grises = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.Canny(grises, 150, 250)
	img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


	height, width, channels = img.shape

	# Detecting objects
	blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

	net.setInput(blob)
	outs = net.forward(output_layers)

	# Showing informations on the screen
	class_ids = []
	confidences = []
	boxes = []
	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > 0.3:
				# Object detected
				center_x = int(detection[0] * width)
				center_y = int(detection[1] * height)
				w = int(detection[2] * width)
				h = int(detection[3] * height)

				# Rectangle coordinates
				x = int(center_x - w / 2)
				y = int(center_y - h / 2)
				# if any(x < 0 for x in [x,y,w,h])
				boxes.append([x, y, w, h])
				confidences.append(float(confidence))
				class_ids.append(class_id)

	
	# ? show predictions image
	indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
	font = cv2.FONT_HERSHEY_PLAIN
	detection_image = img.copy()
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]

			label = str(classes[class_ids[i]])
			color = colors[class_ids[i]]
			# cv2.rectangle(detection_image, (x, y), (x + w, y + h), color, 2)
			# cv2.putText(detection_image, label, (x, y + 30), font, 3, color, 2)

			pt1 = [x,y]
			pt2 = [x + w, y + h]

			if x > 0 and y > 0 and pt2[0] > 0 and pt2[1] > 0:

				cutted_image = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]

				prediction = cnn.prediction('models/ondas-buff.hdf5', cutted_image)[0]
				
				if prediction == 0:
					cv2.rectangle(detection_image, (x, y), (x + w, y + h), (0,255,0), 2)

				else:	
					cv2.rectangle(detection_image, pt1, pt2, risk_colors[prediction], 3)
					cv2.putText(detection_image, risk_text[prediction], (pt1[0],pt1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, risk_colors[prediction], 2)
				

	cv2.imshow("detection_image", detection_image)    
	# cv2.imshow('frame',frame)
	k = cv2.waitKey(1)
	if k == 27:
		break

# cap.release()
cv2.destroyAllWindows()
