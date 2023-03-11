import numpy as np
import tkinter as tk
import cv2 as cv
from PIL import Image, ImageTk

mov = 'images/Tiktok video.mov'
feed = cv.imread('images/Feed.png')
navBar = cv.imread('images/Nav bar.png')
sideBar = cv.imread('images/Sidebar.png')
#crop mov to 393x852
cap = cv.VideoCapture(mov)
# 393x852
cap.set(3, 852)
cap.set(4, 393)

# create window for menu control
window = tk.Tk()
window.title('Menu')
window.geometry('300x300')

while True:
    ret, frame = cap.read()
    if ret:
        # crop frame
        frame = frame[84:852, 0:393]
        print(frame.shape)
        feed[0:852-84, 0:393] = frame
        cv.imshow('frame', feed)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

def faceRecognizer():
    # trained data: trained/model.yml
    # deploy trained data
    model = cv.face.LBPHFaceRecognizer_create()
    model.read('trained/model.yml')
    # load haarcascade
    har = cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    # load video
    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if ret:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = har.detectMultiScale(gray, 1.3, 5)
            for x, y, w, h in faces:
                face = gray[y:y+h, x:x+w]
                face = cv.resize(face, (480, 480))
                label, confidence = model.predict(face)
                print(label, confidence)
                cv.putText(frame, str(label), (x, y), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv.imshow('frame', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

def faceDetector():
    # load haarcascade
    har = cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    # load video
    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if ret:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = har.detectMultiScale(gray, 1.3, 5)
            for x, y, w, h in faces:
                cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv.imshow('frame', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break










