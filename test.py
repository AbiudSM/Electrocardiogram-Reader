import ECG_CNN as cnn
import cv2
import matplotlib.pyplot as plt
import os


# cnn.learning('database/Ondas','ondas.hdf5')

# prediction = cnn.prediction('ondas.hdf5', cv2.imread('database/Ondas/2-onda-T/riesgo26.jpg'))
# print(prediction)

cnn.reinforce_model('database/Ondas','models/ondas-buff.hdf5')