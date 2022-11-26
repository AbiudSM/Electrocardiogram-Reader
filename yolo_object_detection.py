import cv2
import numpy as np
from config import *

def yolo_prediction(img):
    # Load Yolo
    net = cv2.dnn.readNet(YOLO_MODEL, YOLO_CONFIG)

    # Name custom object
    classes = ["ecg"]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    coords = list()

    # img = cv2.resize(img, None, fx=0.4, fy=0.4)
    # img = cv2.resize(img, dsize=(1280,720))

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

    # ? show predictions image
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    detection_image = img.copy()
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]

            coords.append([[x,abs(y)],[x + w, y + h]])
            
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(detection_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(detection_image, label, (x, y + 30), font, 3, color, 2)


    cv2.imshow("detection_image", detection_image)
    key = cv2.waitKey(0)

    cv2.destroyAllWindows()

    return coords, img


img_path = 'src/full/anormal-S.jpg'
img = cv2.imread(img_path)

coords, img = yolo_prediction(img)

for pts in coords:
    try:
        pt1, pt2 = pts
        print(pt1,pt2)
        cutted_image = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]
        cv2.imshow('img',cutted_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)