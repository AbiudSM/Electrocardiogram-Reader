from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import cv2
import imutils
import numpy as np
import pyautogui
import threading

screen_recorded = False

def record_screen():
	global screen_recorded
	global image

	# Label IMAGEN DE SALIDA
	text_output_image = Label(root, text="OUTPUT IMAGE:", font="bold")
	text_output_image.grid(column=1, row=0, padx=5, pady=5)

	screen_recorded = True

	while screen_recorded == True:
		image = np.array(pyautogui.screenshot()) 
		image = cv2.resize(image, (1000,600))

		umbral_1 = first_threshold.get()
		umbral_2 = second_threshold.get()

		grises = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		bordes = cv2.Canny(grises, umbral_1, umbral_2)

		# Para visualizar la imagen en label_output_image en la GUI
		im = Image.fromarray(bordes)
		img = ImageTk.PhotoImage(image=im)
		label_output_image.configure(image=img)
		label_output_image.image = img


	label_output_image.configure(image='')
	label_output_image.image = ''
	text_output_image.grid_forget()


def record_screen_thread():
	threading.Thread(target=record_screen, daemon = True).start()


def stop_button_screen_recording():
	global screen_recorded
	screen_recorded = False


# Root window
root = Tk()
root.geometry('1280x720')
root.title('ECG reader')


label_output_image = Label(root)
label_output_image.grid(column=1, row=1, rowspan=6)


# Thresholds
label_thresholds = Label(root, text="UMBRALES", width=25)
label_thresholds.grid(column=0, row=3, padx=5, pady=5)


first_threshold = Scale(root, from_=0, to=1000, orient=HORIZONTAL)
first_threshold.grid(column=0, row=4)

second_threshold = Scale(root, from_=0, to=1000, orient=HORIZONTAL)
second_threshold.grid(column=0, row=5)


# Buttons
record_screen_button = Button(root, text="Record screen", width=25, command=record_screen_thread)
record_screen_button.grid(column=0, row=0, padx=5, pady=5)

stop_button = Button(root, text="stop_button screeen recording", width=25, command=stop_button_screen_recording)
stop_button.grid(column=0, row=1, padx=5, pady=5)


root.mainloop()