# Electrocardiogram reader
Artificial intelligence applied to read abnormalities present in electrocardiogram images, made up of a convolutional neural network and a YOLOv3 detection model. The CNN is able to classifying electrocardiogram images in order to determine if the wave presented in the image is at risk of having heart problems, assigning the classification for this as healthy or risk status. based on a normal heartbeat, obtaining a response to the state of a patient's heart quickly and without prior knowledge of the risk factors present in the wave. The methodology that was carried out for the development of this project is presented in the article, which consists of a classificatory convolutional neural network, created in the Python programming language and with the help of two of the most important libraries on artificial intelligence, Tensorflow and Keras, characteristics of electrocardiograms and their risk factors with which we can identify when an electrocardiogram is not in good condition are also presented.

## Installation
- Clone the repository
`git clone https://github.com/Robertudo/Electrocardiogram-Reader.git`

- Download YOLOv3 weights model and add it to YOLOv3 folder
[YOLOv3 model](https://drive.google.com/file/d/1-7Jl11emaKUlJ7F8gHq_Ipw2162beOrT/view?usp=share_link "YOLOv3 model")

## Run code
`cd Electrocardiogram-Reader`

`py main.py`

### Interface elements
The detection is handled by the main interface, we can start recording and modify the thresholds and select the detection model.

![interface](https://user-images.githubusercontent.com/71671063/205141232-2cfec2f3-1856-48be-8c79-4b5396567777.jpg)

- Buttons: There are two buttons to handle the device screenshot

- Thresholds: Two of the necessary parameters for this edge detection are the thresholds, the first threshold defines the size of lines will be represented as borders and the second threshold defines the lines that will be represented as edges as long as they are connected to lines with the first threshold represented

- Detection model: There are 3 detection states, disabled, no model is used, YOLOv3 is used for the detection of electrocardiograms and YOLOv3 + CNN is used for the classification between abnormalities presented in the electrocardiogram waves.

**Performance**
OPENCV does not use the GPU in its default usage, the performance obtained with the core i5-12400 processor was stable, it will be working with the GPU application to compare results
![performance](https://user-images.githubusercontent.com/71671063/205141250-7374bea8-acb5-429d-90d3-145d79767c05.jpg)
