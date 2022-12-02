import cv2
import numpy as np
import ECG_CNN as cnn
from config import *

def analyze_image(img):
	"""
	Electrocardiogram full image analyze function
	@returns the image with the ecgs at risk indicated by a rectangle depending on the anomaly
	"""

	# Prediction colors, 0 -> sano, 1 -> onda S, 2 -> onda T, 3 -> onda Q
	risk_colors = [None, (255,0,0), (255,255,0), (0,0,255), ]
	risk_text = ['Sano', 'Onda-S', 'Onda-T', 'Onda-Q']
	
	coords, yolo_image = yolo_prediction(img)

	for pts in coords:
		pt1, pt2 = pts
		cutted_image = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]
		
		prediction = int(cnn.prediction(MODEL_PATH,cutted_image))

		if prediction != 0:
			img = cv2.rectangle(img, pt1, pt2, risk_colors[prediction], 3)
			cv2.putText(img, risk_text[prediction], (pt1[0],pt1[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, risk_colors[prediction], 2)

	return img


def yolo_prediction(image):
	"""
	YOLOv3 prediction of image, detect all electrocardiograms in the image
	@returns a list (coords,image): with the coordinates where the detected ECGs are located and the input image with rectangles where the ECGs are located
	"""

	img = image.copy()

	# Load Yolo
	net = cv2.dnn.readNet(YOLO_MODEL, YOLO_CONFIG)


	layer_names = net.getLayerNames()
	output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

	coords = list()

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

				boxes.append([x, y, w, h])
				confidences.append(float(confidence))
				class_ids.append(class_id)

	indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			
			pt1 = [x,y]
			pt2 = [x + w, y + h]

			# ? if all corrds are positive
			if x > 0 and y > 0 and pt2[0] > 0 and pt2[1] > 0:
				coords.append([pt1,pt2])
				cv2.rectangle(img, pt1, pt2, (0,255,0), 2)

	return coords, img