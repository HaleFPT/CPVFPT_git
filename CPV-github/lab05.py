# Feature base alignment

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def take_images():
    cap = cv.VideoCapture(0)
    captured = 0
    # if captured = 2 break
    while True:
        ret, frame = cap.read()
        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        if cv.waitKey(1) & 0xFF == ord('c'):
            captured += 1
            cv.imwrite('img' + str(captured) + '.jpg', frame)
            print('Image ' + str(captured) + ' captured')
        if captured == 2:
            break
    cap.release()
    cv.destroyAllWindows()

def preprocess_images():
    img1 = cv.imread('images/img1.jpg')
    img2 = cv.imread('images/img2.jpg')
    img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    return img1, img2

def feature_base_alignment(img1,img2):
    # Initiate ORB detector
    orb = cv.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # good matches
    good = int(len(matches)*0.15)
    matches = matches[:good]
    # Draw first 10 matches.
    img3 = cv.drawMatches(img1,kp1,img2,kp2,matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()
    # Extract the location of the matching keypoints in both images
    pts1 = []
    pts2 = []
    for match in matches:
        pts1.append(kp1[match.queryIdx].pt)
        pts2.append(kp2[match.trainIdx].pt)
    print(len(pts1))
    print(len(pts2))
    # Estimate the homography matrix using RANSAC algorithm
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    M, mask = cv.estimateAffinePartial2D(pts1, pts2, method=cv.RANSAC, ransacReprojThreshold=5.0)
    # Use the homography matrix to align the images
    height, width = img2.shape
    aligned_img = cv.warpAffine(img1, M, (img2.shape[1], img2.shape[0]))
    return aligned_img

def main():
    # take_images()
    img1, img2 = preprocess_images()
    dst = feature_base_alignment(img1,img2)
    plt.imshow(dst)
    plt.show()

if __name__ == '__main__':
    main()