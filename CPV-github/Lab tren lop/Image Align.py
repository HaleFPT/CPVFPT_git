import cv2 as cv
import numpy as np


def image_alignment(imgRef, imgTest, bshow=True):
    """
    https://thinkinfi.com/image-alignment-and-registration-with-opencv/
    Dùng feature matching để tìm ra ma trận biến đổi
    góc nhìn, sau đó biến đổi ngược từ imgTest về imgRef
    :param img1:
    :param img2:
    :return:
    """
    # Convert to grayscale.
    imgTest_grey = cv.cvtColor(imgTest, cv.COLOR_BGR2GRAY)
    imgRef_grey = cv.cvtColor(imgRef, cv.COLOR_BGR2GRAY)
    height, width = imgRef_grey.shape

    # Configure ORB feature detector Algorithm with 1000 features.
    orb_detector = cv.ORB_create(1000)

    # Extract key points and descriptors for both images
    keyPoint1, des1 = orb_detector.detectAndCompute(imgTest_grey, None)
    keyPoint2, des2 = orb_detector.detectAndCompute(imgRef_grey, None)

    # Display keypoints for reference image in green color
    imgKp_Ref = cv.drawKeypoints(imgRef, keyPoint1, 0, (0, 222, 0), None)
    imgKp_Ref = cv.resize(imgKp_Ref, (width // 2, height // 2))

    cv.imshow('Key Points', imgKp_Ref)
    cv.waitKey(0)

    # Match features between two images using Brute Force matcher with Hamming distance
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # Match the two sets of descriptors.
    matches = list(matcher.match(des1, des2))

    # Sort matches on the basis of their Hamming distance.
    matches.sort(key=lambda x: x.distance)
    print(matches)
    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches) * 0.9)]
    no_of_matches = len(matches)

    # Display only 100 best matches {good[:100}
    imgMatch = cv.drawMatches(imgTest, keyPoint2, imgRef, keyPoint1, matches[:20], None, flags=2)
    imgMatch = cv.resize(imgMatch, (width // 3, height // 3))

    cv.imshow('Image Match', imgMatch)
    cv.waitKey(0)

    # Define 2x2 empty matrices
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    # Storing values to the matrices
    for i in range(len(matches)):
        p1[i, :] = keyPoint1[matches[i].queryIdx].pt
        p2[i, :] = keyPoint2[matches[i].trainIdx].pt
    # Find the homography matrix.
    homography, mask = cv.findHomography(p1, p2, cv.RANSAC)

    # Use homography matrix to transform the unaligned image wrt the reference image.
    aligned_img = cv.warpPerspective(imgTest, homography, (width, height))
    # Resizing the image to display in our screen (optional)
    aligned_img = cv.resize(aligned_img, (width, height))

    # Copy of input image
    # imgTest_cp = imgTest.copy()
    # imgTest_cp = cv.resize(imgTest_cp, (width // 3, height // 3))
    # Save the align image output.
    # cv2.imwrite('output.jpg', aligned_img)

    # cv.imshow('Input Image', imgTest_cp)
    cv.namedWindow('Output Image', cv.WINDOW_NORMAL)
    cv.imshow('Output Image', aligned_img)
    cv.waitKey(0)


def image_alignment_demo():
    """
    Dùng camera để demo cái này
    :return:
    """
    cap = cv.VideoCapture(0)
    imgRef = None
    imgTest = None
    cnt = 0
    while (1):
        ret, frame = cap.read()
        if ret:
            cv.imshow("vid", frame)
            k = cv.waitKey(10)
            if k % 256 == 32:
                # Space pressed
                if cnt == 0:
                    imgRef = frame
                    cv.imshow("imgRef", imgRef)
                    cv.waitKey(5)
                    cnt += 1
                elif cnt == 1:
                    imgTest = frame
                    cv.imshow("imgTest", imgTest)
                    cnt += 1
                if cnt == 2: break
        else:
            break
    cap.release()
    # if imgRef != None and imgTest != None:
    image_alignment(imgRef, imgTest)


def main():
    image_alignment_demo()
    cv.destroyAllWindows()


main()
