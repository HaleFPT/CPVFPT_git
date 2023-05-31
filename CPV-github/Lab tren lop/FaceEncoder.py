import cv2 as cv
import pathlib as pl
import numpy as np


# load faces ('Faces') and take the name
har = cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')
Path = 'Faces'
path = pl.Path(Path)
files = path.glob('*.jpg')
# save faces in processed images (imwrite)
for file in files:
    img = cv.imread(str(file))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = har.detectMultiScale(gray, 1.3, 5)
    for x, y, w, h in faces:
        face = gray[y:y+h, x:x+w]
        face = cv.resize(face, (480, 480))
        cv.imwrite('processed image/' + str(file).split('\\')[1], face)
        print(str(file))
