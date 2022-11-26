import cv2
import matplotlib.pyplot as plt

input_img=cv2.imread('dataset/sano/ecg3.jpg')
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
input_img_resize = cv2.resize(input_img,(51,51))

cv2.imwrite('test.jpg',input_img)