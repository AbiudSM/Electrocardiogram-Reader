import cv2
import numpy as np
import ECG_CNN as cnn

# Electrocardiogram full image analyze_image function
# ? returns the image with the ecgs at risk indicated by a red rectangle
def analyze_image(model_path, img):

    risk_image = False
        
    # Prediction colors, 0 -> sano, 1 -> onda S, 2 -> onda T, 3 -> onda Q
    risk_colors = [None, (255,0,0), (255,255,0), (0,0,255), ]
    risk_text = ['Sano', 'Onda-S', 'Onda-T', 'Onda-Q']
    
    # img = cv2.resize(img, (width,height), interpolation = cv2.INTER_AREA)

    coords, img = yolo_prediction(img)

    for pts in coords:
        pt1, pt2 = pts
        cutted_image = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]
        
        prediction = int(cnn.prediction(model_path,cutted_image))

        if prediction != 0:
            risk_image = cv2.rectangle(img, pt1, pt2, risk_colors[prediction], 3)
            cv2.putText(img, risk_text[prediction], (pt1[0],pt1[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, risk_colors[prediction], 2)

    return risk_image


def yolo_prediction(img):
    # Load Yolo
    net = cv2.dnn.readNet("yolov3/yolov3_training_3000.weights", "yolov3/yolov3_testing.cfg")

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
    # detection_image = img.copy()
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]

            coords.append([[x,abs(y)],[x + w, y + h]])
            
            # label = str(classes[class_ids[i]])
            # color = colors[class_ids[i]]
            # cv2.rectangle(detection_image, (x, y), (x + w, y + h), color, 2)
            # cv2.putText(detection_image, label, (x, y + 30), font, 3, color, 2)


    # cv2.imshow("detection_image", detection_image)
    # key = cv2.waitKey(0)

    # cv2.destroyAllWindows()

    return coords, img


risk_image = analyze_image('models/ondas-buff.hdf5', cv2.imread('src/full/anormal-T-cut.jpg'))
if type(risk_image) == bool:
    print('\n\nElectrocardiograma completamenete sano')
else:
    cv2.imshow('risk_image',risk_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()