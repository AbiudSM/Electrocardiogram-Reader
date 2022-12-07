import os
import cv2
import numpy as np
from config import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.core import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.models import load_model
K.set_image_data_format('channels_last')


# Variables
num_classes = 0
labels_t=[]
img_data=[]
input_shape=0
model=0
X_train=''
X_val=''
y_train=''
y_val=''

def load_dataset(data_path):
    global labels_t
    global img_data
    global num_classes

    img_data_list=[]
    n_imag=0
    n_imag_array=[]
    n_imag_cont=0

    print('Loading Dataset from directory: ' + data_path)

    data_dir_list= os.listdir(data_path)
    num_classes = len(data_dir_list)
    for dataset in data_dir_list:
        img_list = os.listdir(data_path + '/' + dataset)
        print(f'Loading image from: {dataset}\n')
        for img in img_list:
            n_imag+=1
            input_img=cv2.imread(data_path + '/' + dataset + '/' + img)
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            input_img_resize = cv2.resize(input_img,(IMG_ROWS,IMG_COLS))
            img_data_list.append(input_img_resize)
        n_imag_array.append(n_imag)
        n_imag_cont+=1

    img_data = np.array(img_data_list)
    img_data = img_data.astype('float32')
    img_data /= 255
    img_data = np.expand_dims(img_data, axis=3)

    # Set labels
    num_of_samples = img_data.shape[0]
    labels = np.ones((num_of_samples),dtype='int64')

    labels[0:n_imag_array[0]]=0
    for i in range(len(n_imag_array)-1):
        labels[n_imag_array[i]:n_imag_array[i+1]] = i+1

    # labels[0:n_imag_array[0]]=0
    # labels[n_imag_array[0]:n_imag_array[1]]=1

    labels_t=labels

def set_image_labels(labels_t,num_classes):
    global X_train, X_val, y_train, y_val
    global input_shape
    
    print(f'{len(labels_t)} images uploaded')

    # Convert class labels
    Y = np_utils.to_categorical(labels_t,num_classes)

    x,y = shuffle(img_data,Y,random_state=2)
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=2)

    # ? CNN input shape = image size
    input_shape=img_data[0].shape
    
    print(f'{len(x)} vectors \n{len(y)} labels')


def set_model(input_shape):
    global model

    # CNN structure
    model = Sequential()
    model.add(Conv2D(32,(3,3),padding='same',input_shape=input_shape))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(DROPOUT))
    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(DROPOUT))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    
def train_model(model: Sequential, model_name: str, epochs:int = NUM_EPOCH):
    global X_train, X_val, y_train, y_val

    # Training model
    print('Training model...')
    
    model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER,metrics=["accuracy"])
    tbCallBack = callbacks.TensorBoard(log_dir='.\\log\\'+OPTIMIZER+'-gpu-todo', histogram_freq=1,write_graph=True, write_images=False)
    tbCallBack.set_model(model)

    while True:
        model.fit(X_train, y_train, batch_size=320, epochs=epochs, verbose=1, validation_data=(X_val, y_val),callbacks=[tbCallBack])
        
        score = model.evaluate(X_train, y_train)
        print(f'loss/accuracy: {score}')

        if score[1] > MAX_SCORE:
            break

    # Save and load model and weights
    print('Saving model...')
    model.save(model_name)
    load_model(model_name)
    print('Model saved!')


def reinforce_model(data_path: str, model_path: str, epochs:int = NUM_EPOCH) -> None:
    """
    Train a previously saved model.
    @data_path: folder path containing images for the training
    @model_path: previously saved model path
    @epochs: number of epochs for the training process
    """
    load_dataset(data_path)
    set_image_labels(labels_t,num_classes)
    model = load_model(model_path)
    train_model(model,model_path, epochs)

def learning(data_path: str, model_name: str):
    """
    CNN learning process, load dataset and train the model.
    @data_path: folder path containing images for the training
    @model_name: model output filename in .hdf5 format
    """
    load_dataset(data_path)
    set_image_labels(labels_t,num_classes)
    set_model(input_shape)
    train_model(model,model_name)

def prediction(model_path: str, image: np.ndarray) -> int:
    """
    Get image prediction class number
    @model_path: string of previously saved model path
    @image: image read by cv2 module
    """

    
    num_channel=1

    model = load_model(model_path)

    test_image = image
    try:
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    except:
        pass
    test_image = cv2.resize(test_image,(IMG_ROWS,IMG_COLS))
    test_image = np.array(test_image)
    test_image = test_image.astype('float32')
    test_image /= 255

    if num_channel == 1:
        if K.set_image_data_format('channels_last') == 'th':
            test_image = np.expand_dims(test_image, axis=0)
            test_image = np.expand_dims(test_image, axis=0)
        else:
            test_image = np.expand_dims(test_image, axis=2)
            test_image = np.expand_dims(test_image, axis=0)
    else:
        if K.set_image_data_format('channels_last') == 'th':
            test_image = np.rollaxis(test_image,2,0)
            test_image = np.expand_dims(test_image, axis=0)
        else:
            test_image = np.expand_dims(test_image, axis=0)

    # print(f'\n\n{(model.predict(test_image) > 0.5).astype("int32")}\n\n')
    return model.predict_classes(test_image)