from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import cv2
import imutils
import numpy as np

global_path_image = ''

def elegir_imagen():
    # Especificar los tipos de archivos, para elegir solo a las imágenes
    path_image = filedialog.askopenfilename(filetypes = [
        ("image", ".jpeg"),
        ("image", ".png"),
        ("image", ".jpg")])

    global global_path_image
    global_path_image = path_image

    if len(path_image) > 0:
        global image

        # Leer la imagen de entrada y la redimensionamos
        image = cv2.imread(path_image)
        image= imutils.resize(image, height=380)

        # Para visualizar la imagen de entrada en la GUI
        imageToShow= imutils.resize(image, width=180)
        imageToShow = cv2.cvtColor(imageToShow, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(imageToShow )
        img = ImageTk.PhotoImage(image=im)

        lblInputImage.configure(image=img)
        lblInputImage.image = img

        # Label IMAGEN DE ENTRADA
        lblInfo1 = Label(root, text="IMAGEN DE ENTRADA:")
        lblInfo1.grid(column=0, row=1, padx=5, pady=5)

        # Al momento que leemos la imagen de entrada, vaciamos
        # la iamgen de salida y se limpia la selección de los
        # radiobutton
        lblOutputImage.image = ""
        selected.set(0)

def obtener_umbral(values):
    global image
    global bordes

    umbral_1 = primer_umbral.get()
    umbral_2 = segundo_umbral.get()

    grises = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(grises, umbral_1, umbral_2)

    # Para visualizar la imagen en lblOutputImage en la GUI
    im = Image.fromarray(bordes)
    img = ImageTk.PhotoImage(image=im)
    lblOutputImage.configure(image=img)
    lblOutputImage.image = img

    # Label IMAGEN DE SALIDA
    lblInfo3 = Label(root, text="IMAGEN DE SALIDA:", font="bold")
    lblInfo3.grid(column=1, row=0, padx=5, pady=5)

    # Guardar imagen button
    save = Button(root, text="Guardar imagen", width=25, command=save_image)
    save.grid(column=1, row=10, padx=5, pady=5)

def save_image():
    global bordes
    global global_path_image


    if len(global_path_image) > 0:
        cv2.imwrite(global_path_image, bordes)

# root window
root = Tk()
root.geometry('800x700')
root.title('ECG reader')

# Label donde se presentará la imagen de entrada
lblInputImage = Label(root)
lblInputImage.grid(column=0, row=2)
# Label donde se presentará la imagen de salida
lblOutputImage = Label(root)
lblOutputImage.grid(column=1, row=1, rowspan=6)


# Label Umbrales
lblInfo2 = Label(root, text="UMBRALES", width=25)
lblInfo2.grid(column=0, row=3, padx=5, pady=5)

# Label(root, text="Primer Umbral").pack()
primer_umbral = Scale(root, from_=0, to=1000, orient=HORIZONTAL, command=obtener_umbral)

# Label(root, text="Segundo Umbral").pack()
segundo_umbral = Scale(root, from_=0, to=1000, orient=HORIZONTAL, command=obtener_umbral)


# Creamos los radio buttons y la ubicación que estos ocuparán
selected = IntVar()
primer_umbral.grid(column=0, row=4)
segundo_umbral.grid(column=0, row=5)

# Creamos el botón para elegir la imagen de entrada
btn = Button(root, text="Elegir imagen", width=25, command=elegir_imagen)
btn.grid(column=0, row=0, padx=5, pady=5)


# umbral_button = Button(root, text='Continuar!', command=obtener_umbral).grid(column=0, row=6, padx=5, pady=25)

root.mainloop()