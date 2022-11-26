import cv2, os
import numpy as np
from tkinter import filedialog
from ECG_CNN import prediction
import matplotlib.pyplot as plt

# Function to obtain the necessary thresholds for an individual electrocardiogram wave
# ? returns an image with Canny applied
def ecg_generator(model, image):
    ecg = 'false'
    threshold_1 = 0
    threshold_2 = 100

    while True:
        grises = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bordes = cv2.Canny(grises, threshold_1, threshold_2)

        result = prediction(model, bordes)

        threshold_1 += 50
        threshold_2 += 50

        if threshold_1 > 1000:
            break

        if result == 1:
            ecg = bordes
            break
    
    return ecg