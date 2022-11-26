import ECG_CNN as cnn
import cv2

# Electrocardiogram full image analysis function
# ? returns the image with the ecgs at risk indicated by a red rectangle
def analisis(model_path, img, ecg_rows, ecg_cols):
    width = 1920
    height = 1080

    risk_image = False
        
    x_distance = int(width / ecg_cols)
    y_distance = int(height / ecg_rows)
    
    pt1 = [0,0]
    pt2 = [x_distance,y_distance]

    # Colores por prediccion de onda, 0 -> sano, 1 -> onda S, 2 -> onda T, 3 -> onda Q
    risk_colors = [None, (255,0,0), (255,255,0), (0,0,255), ]
    risk_text = ['Sano', 'Onda-S', 'Onda-T', 'Onda-Q']

    img = cv2.resize(img, (width,height), interpolation = cv2.INTER_AREA)

    for _ in range (ecg_rows):
        for _ in range(ecg_cols):
            cutted_image = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]
            prediction = int(cnn.prediction(model_path,cutted_image))

            if prediction != 0:
                risk_image = cv2.rectangle(img, pt1, pt2, risk_colors[prediction], 3)
                cv2.putText(img, risk_text[prediction], (pt1[0],pt1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, risk_colors[prediction], 2)

            pt1[0] += x_distance
            pt2[0] += x_distance
        
        pt1[0] = 0
        pt1[1] += y_distance

        pt2[0] = x_distance
        pt2[1] += y_distance

    return risk_image

import matplotlib.pyplot as plt
risk_image = analisis('models/ondas-buff.hdf5', cv2.imread('src/full/edges.jpg'), 3, 7)

if type(risk_image) == bool:
    print('Electrocardiograma completamenete sano')
else:
    plt.imshow(risk_image)
    plt.show()
    