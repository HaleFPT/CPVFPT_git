import cv2 as cv
import numpy as np


cv.setUseOptimized(True)
print(cv.getBuildInformation())
drawing = False
img = np.ones((600,800,3),np.uint8)*255
current_rectangle = []


class Rectangle:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def get_width(self):
        return abs(self.x2 - self.x1)

    def get_height(self):
        return abs(self.y2 - self.y1)

    def get_center(self):
        return (int((self.x1 + self.x2) / 2), int((self.y1 + self.y2) / 2))

    def fill_rectangle(self, img, color):
        cv.rectangle(img, (self.x1, self.y1), (self.x2, self.y2), color, -1)

def white_background(img):
    img[:] = (255, 255, 255)

def draw_rectangle(event, x, y, flags, param):
    global drawing, img, current_rectangle, frame_copy
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        current_rectangle = Rectangle(x, y, x, y)
        frame_copy = img.copy()
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            current_rectangle.x2 = x
            current_rectangle.y2 = y
            #fill rectangle while delete old filled rectangle
            white_background(img)
            cv.rectangle(img, (current_rectangle.x1, current_rectangle.y1), (current_rectangle.x2, current_rectangle.y2), (0, 0, 255), 1)

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        if current_rectangle is not None and current_rectangle.get_width() > 0 and current_rectangle.get_height() > 0:
            current_rectangle.fill_rectangle(img, (0, 0, 255))
            current_rectangle = None
        frame_copy = None
    elif current_rectangle is not None:
        img = frame_copy.copy()
        cv.rectangle(img, (current_rectangle.x1, current_rectangle.y1), (current_rectangle.x2, current_rectangle.y2), (0, 0, 255), 1)
        cv.imshow("Frame", img)

def main():
    global img
    cv.namedWindow("Frame")
    cv.setMouseCallback("Frame", draw_rectangle)
    while True:
        cv.imshow("Frame", img)
        key = cv.waitKey(1)
        if key == 27:
            break
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()

