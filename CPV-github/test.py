import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk

def capture():
    print("capture")

def training():
    print("training")

def recognizer():
    print("recognizer")

# create menu with buttons using tkinter
root = tk.Tk()
root.title("Face Recognition System")
root.geometry("300x300")
root.resizable(False, False)
capture_button = tk.Button(root, text="Capture", command=capture)
training_button = tk.Button(root, text="Training", command=training)
recognizer_button = tk.Button(root, text="Recognizer", command=recognizer)
capture_button.pack()
training_button.pack()
recognizer_button.pack()
root.mainloop()




