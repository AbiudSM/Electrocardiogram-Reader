from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import cv2
import imutils
import numpy as np
import pyautogui
import threading
import methods

screen_recorded = False
detection_status = 0

def record_screen():
	global screen_recorded
	global image
	global detection_status

	record_screen_button.config(state='disabled')
	stop_button.config(state='normal')

	# Label output image
	text_output_image = Label(root, text="OUTPUT IMAGE:", font="bold")
	text_output_image.grid(column=1, row=0, padx=5, pady=5)

	screen_recorded = True

	while screen_recorded == True:
		image = np.array(pyautogui.screenshot())
		image = cv2.resize(image, (1000,600))

		threshold_1 = first_threshold.get()
		threshold_2 = second_threshold.get()

		grises = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		edges = cv2.Canny(grises, threshold_1, threshold_2)

		if detection_status == 1:
			edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
			coords, edges = methods.yolo_prediction(edges)

		elif detection_status == 2:
			edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
			edges = methods.analyze_image(edges)

		# Set image to output image label
		im = Image.fromarray(edges)
		img = ImageTk.PhotoImage(image=im)
		label_output_image.configure(image=img)
		label_output_image.image = img


	label_output_image.configure(image='')
	label_output_image.image = ''
	text_output_image.grid_forget()
	record_screen_button.config(state='normal')
	stop_button.config(state='disabled')


def record_screen_thread():
	threading.Thread(target=record_screen, daemon = True).start()


def stop_button_screen_recording():
	global screen_recorded
	screen_recorded = False


def set_detection_status():
	global detection_status
	detection_status = selection.get()



# Root window
root = Tk()
root.geometry('1280x720')
root.title('ECG reader')

# Buttons
record_screen_button = Button(root, text="Record screen", width=25, command=record_screen_thread)
record_screen_button.grid(column=0, row=0, padx=5, pady=5)

stop_button = Button(root, text="stop_button screeen recording", width=25, command=stop_button_screen_recording, state='disabled')
stop_button.grid(column=0, row=1, padx=5, pady=5)


# Thresholds
label_thresholds = Label(root, text="UMBRALES", width=25)
label_thresholds.grid(column=0, row=3, padx=5, pady=5)


first_threshold = Scale(root, from_=0, to=500, orient=HORIZONTAL, length=200)
first_threshold.grid(column=0, row=4)

second_threshold = Scale(root, from_=0, to=500, orient=HORIZONTAL, length=200)
second_threshold.grid(column=0, row=5)


# Radio buttons
selection = IntVar()
Radiobutton(root, text="Desactivado", variable=selection, value=0, command=set_detection_status).grid(column=0, row=6)
Radiobutton(root, text="YOLOv3", variable=selection, value=1, command=set_detection_status).grid(column=0, row=7)
Radiobutton(root, text="YOLOv3 + CNN", variable=selection, value=2, command=set_detection_status).grid(column=0, row=8)


# Output image
label_output_image = Label(root)
label_output_image.grid(column=1, row=1, rowspan=6)

root.mainloop()