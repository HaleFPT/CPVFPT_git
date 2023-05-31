# image stiching

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob

# load images
imgdir = 'panorama/*.jpg'
images = [cv.imread(file) for file in glob.glob(imgdir)]

# stitcher
stitcher = cv.Stitcher.create()
(status, stitched) = stitcher.stitch(images)

# show result
if status == 0:
    cv.imshow('Stitched', stitched)
    cv.waitKey(0)
else:
    print('Error during stitching')

# save result
cv.imwrite('panorama/stitched.jpg', stitched)

# close all windows
cv.destroyAllWindows()