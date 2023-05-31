import cv2 as cv
from cv2 import cuda
import numpy as np
from cv2 import HOGDescriptor_getDefaultPeopleDetector

# Function 1: Harris Corner Detector for feature detection
def harris_corner_detector(image, block_size=2, ksize=3, k=0.04):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, block_size, ksize, k)
    dst = cv.dilate(dst,None)
    image[dst>0.01*dst.max()]=[0,0,255] # mark the corners on the original image
    return image

# Function 2: HOG feature descriptor
def HOG(frame):
    # create HOG descriptor
    hog = cv.HOGDescriptor()
    # set the SVMDetector
    hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
    # detect people in the image
    (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return frame


# Function 3: Canny Edge Detection
# Function to perform Canny edge detection using GPU acceleration
def Canny_edge_detection(image, minVal=100, maxVal=200):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, minVal, maxVal)
    return edges


# Function 4: Hough Transform for Line Detection
def Hough_transform(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    lines = cv.HoughLines(edges, 1, np.pi/180, 200)
    if lines is not None:
        for rho,theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv.line(image,(x1,y1),(x2,y2),(0,0,255),2)
    return image

file = 'Tiktok video.mov'
# show loop video
cap = cv.VideoCapture(file)
while True:
    ret, frame = cap.read()
    frame = harris_corner_detector(frame)
    #frame = HOG(frame)
    #frame = Canny_edge_detection(frame)
    #frame = Hough_transform(frame)
    if ret:
        cv.imshow('frame', frame)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break


cap.release()
cv.destroyAllWindows()
