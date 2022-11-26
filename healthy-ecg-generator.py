"""
Generate individual 50x50 ECG images in a healthy state
based on erroneous predictions of the model to ECG in healthy state
"""

import ECG_CNN as cnn
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import time



def save_healthy_ecg(model_path, img, ecg_rows, ecg_cols):
    width = 1920
    height = 1080
    x_distance = int(width / ecg_cols)
    y_distance = int(height / ecg_rows)
    pt1 = [0,0]
    pt2 = [x_distance,y_distance]
    img = cv2.resize(img, (width,height), interpolation = cv2.INTER_AREA)

    for _ in range (ecg_rows):
        for _ in range(ecg_cols):
            cutted_image = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]
            prediction = int(cnn.prediction(model_path,cutted_image))

            if prediction != 0:
                date = datetime.now().strftime("%d-%H-%M-%S")
                cv2.imwrite(f'test/generador-sano/sano-{date}.jpg',cutted_image)
                time.sleep(1)

            pt1[0] += x_distance; pt2[0] += x_distance
        
        pt1[0] = 0; pt1[1] += y_distance; pt2[0] = x_distance; pt2[1] += y_distance


risk_image = save_healthy_ecg('models/ondas.hdf5', cv2.imread('src/full/edges.jpg'), 3, 7)