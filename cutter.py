import cv2
import matplotlib.pyplot as plt

# ? returns a list of images of each electrocardiogram
def get_ecg_list(img, ecg_rows, ecg_cols):

    width = 1920
    height = 1080
    
    images = list()
    
    x_distance = int(width / ecg_cols)
    y_distance = int(height / ecg_rows)
    
    pt1 = [0,0]
    pt2 = [x_distance,y_distance]
    color = (0,255,0)

    img = cv2.resize(img, (width,height), interpolation = cv2.INTER_AREA)

    for _ in range (ecg_rows):
        for _ in range(ecg_cols):
            roi = cv2.rectangle(img, pt1, pt2, color, 3)
            cutted_image = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]
            cutted_image = cv2.resize(cutted_image, (150,150), interpolation = cv2.INTER_AREA)
            images.append(cutted_image)
            pt1[0] = pt2[0]
            pt2[0] += x_distance
        
        pt1[0] = 0
        pt1[1] += y_distance

        pt2[0] = x_distance
        pt2[1] += y_distance

    plt.imshow(roi)
    plt.show()
    return images

img = cv2.imread('full/full_ecg.jpg')
images = get_ecg_list(img, 3, 15)

for result in images:
    plt.imshow(result)
    plt.show()