#RANSAC feature matching and alignment

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import sys
import time


def RANSAC_feature_matching(img_captured, img_changed):
    # Read the images
    img = cv.imread(img_captured)
    img_changed = cv.imread(img_changed)
    img = cv.GaussianBlur(img, (3, 3), 0)
    img_changed = cv.GaussianBlur(img_changed, (3, 3), 0)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_changed = cv.cvtColor(img_changed, cv.COLOR_BGR2RGB)


    # Convert to grayscale
    img_grey = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    height, width= img_grey.shape
    img_changed_grey = cv.cvtColor(img_changed, cv.COLOR_RGB2GRAY)

    # Configure ORB feature detector Algorithm
    orb_detector = cv.ORB_create(1000)

    # Extract key points and descriptors for both images
    keyPoint1, des1 = orb_detector.detectAndCompute(img_grey, None)
    keyPoint2, des2 = orb_detector.detectAndCompute(img_changed_grey, None)

    # Display keypoints for reference image in green color
    imgKp_Ref = cv.drawKeypoints(img, keyPoint1, 0, (0, 222, 0), None)
    imgKp_Ref = cv.resize(imgKp_Ref, (img.shape[1] // 2, img.shape[0] // 2))

    # Display keypoints for changed image in red color
    imgKp_Changed = cv.drawKeypoints(img_changed, keyPoint2, 0, (222, 0, 0), None)
    imgKp_Changed = cv.resize(imgKp_Changed, (img.shape[1] // 2, img.shape[0] // 2))

    # Display both images
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(imgKp_Ref)
    plt.title('Reference Image')
    plt.subplot(1, 2, 2)
    plt.imshow(imgKp_Changed)
    plt.title('Changed Image')
    plt.show()

    # Match features between two images using Brute Force matcher with Hamming distance
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # Match the two sets of descriptors.
    matches = list(matcher.match(des1, des2))

    # Sort matches on the basis of their Hamming distance.
    matches.sort(key=lambda x: x.distance)

    # take the top 75% matches forward.
    matches = matches[:int(len(matches) * 0.9)]
    no_of_matches = len(matches)

    #display the top 100 matches
    img_matches = cv.drawMatches(img, keyPoint1, img_changed, keyPoint2, matches[:100], None, flags=2)
    plt.figure(figsize=(20, 10))
    plt.imshow(img_matches)
    plt.title('Top 100 matches')
    plt.show()


    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = keyPoint1[matches[i].queryIdx].pt
        p2[i, :] = keyPoint2[matches[i].trainIdx].pt

    # Find the homography matrix.
    homography, mask = cv.findHomography(p1, p2, cv.RANSAC)

    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_img = cv.warpPerspective(img, homography, (width, height))

    # Shơw the output.
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Reference Image')
    plt.subplot(1, 2, 2)
    plt.imshow(transformed_img)
    plt.title('Changed Image')
    plt.show()

def main():
    cam = cv.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        cv.imshow('Camera', frame)

        if cv.waitKey(1) & 0xFF == ord('c'):
            img_capture = frame
            cv.imwrite('img_capture.jpg', img_capture)
            print('Image captured')
            break

    while True:
        ret, frame = cam.read()
        cv.imshow('Camera', frame)

        if cv.waitKey(1) & 0xFF == ord('c'):
            img_changed = frame
            cv.imwrite('img_changed.jpg', img_changed)
            print('Image changed')
            break

    cam.release()
    cv.destroyAllWindows()

    RANSAC_feature_matching('img_capture.jpg', 'img_changed.jpg')

if __name__ == '__main__':
    main()
