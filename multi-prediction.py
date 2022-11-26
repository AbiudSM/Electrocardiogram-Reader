from ECG_CNN import *
import random

# learning('database','test.hdf5')

i = random.randint(1,100)
img = f'sano{i}.jpg'
image = cv2.imread(f'database/sano/{img}')
pred = prediction('test.hdf5',image)

print(f'Imagen: {img}\nPrediccion: {pred}')