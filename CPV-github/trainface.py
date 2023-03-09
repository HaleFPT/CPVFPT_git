import cv2 as cv
import numpy as np
import pathlib as pl

model = cv.face.LBPHFaceRecognizer_create()
har = cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')
Path = 'processed image'
# train faces in processed images ('processed image') and save the model
path = pl.Path(Path)
files = path.glob('*.jpg')
faceData = []
names = []
index_labels = []








