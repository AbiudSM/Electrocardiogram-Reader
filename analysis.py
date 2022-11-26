import ECG_CNN as cnn
import cv2

# Electrocardiogram full image analysis function
# ? returns the image with the ecgs at risk indicated by a red rectangle
def analysis(model_path, img, ecg_rows, ecg_cols):
    width = 1920
    height = 1080

    risk_image = False
        
    x_distance = int(width / ecg_cols)
    y_distance = int(height / ecg_rows)
    
    pt1 = [0,0]
    pt2 = [x_distance,y_distance]
    red = (255,0,0)

    img = cv2.resize(img, (width,height), interpolation = cv2.INTER_AREA)

    for _ in range (ecg_rows):
        for _ in range(ecg_cols):
            cutted_image = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]
            prediction = cnn.prediction(model_path,cutted_image)
            if prediction == 0:
                risk_image = cv2.rectangle(img, pt1, pt2, red, 3)

            pt1[0] += x_distance
            pt2[0] += x_distance
        
        pt1[0] = 0
        pt1[1] += y_distance

        pt2[0] = x_distance
        pt2[1] += y_distance

    return risk_image

import matplotlib.pyplot as plt
risk_image = analysis('perfect.hdf5', cv2.imread('full/full.jpg'), 3, 14)
if risk_image.any():
    plt.imshow(risk_image)
    plt.show()
    