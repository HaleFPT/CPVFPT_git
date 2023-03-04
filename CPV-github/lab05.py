import cv2 as cv
import numpy as np


def image_alignment(imgRef, imgTest):
    # Convert the images to grayscale.
    imgRefGray = cv.cvtColor(imgRef, cv.COLOR_BGR2GRAY)
    imgTestGray = cv.cvtColor(imgTest, cv.COLOR_BGR2GRAY)

    # Detect the keypoints and descriptors using SIFT.
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(imgRefGray, None)
    kp2, des2 = sift.detectAndCompute(imgTestGray, None)

    # Match the keypoints using KNN matcher.
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(des1, des2, 2)

    # Filter matches using the Lowe's ratio test.
    ratio_thresh = 0.75
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    # Draw the matches.
    img_matches = np.empty((max(imgRef.shape[0], imgTest.shape[0]), imgRef.shape[1] + imgTest.shape[1], 3),
                           dtype=np.uint8)
    cv.drawMatches(imgRef, kp1, imgTest, kp2, good_matches, img_matches,
                   flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Localize the object.
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find the homography matrix.
    homography, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC)

    # Use homography matrix to transform the unaligned image wrt the reference image.
    height, width, channels = imgRef.shape
    aligned_img = cv.warpPerspective(imgTest, homography, (width, height))
    # Resizing the image to display in our screen (optional)
    aligned_img = cv.resize(aligned_img, (width, height))

    return aligned_img


def main():
    button_pressed = False
    ref_img = None
    test_img = None

    cam = cv.VideoCapture(0)
    cv.namedWindow("Camera")
    cv.namedWindow("Aligned")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv.imshow("Camera", frame)

        k = cv.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

        elif k % 256 == 32:
            # SPACE pressed
            button_pressed = not button_pressed
            if button_pressed:
                ref_img = frame.copy()
                print("Reference image captured")
            else:
                test_img = frame.copy()
                print("Test image captured")

                # Align the test image with the reference image.
                aligned_img = image_alignment(ref_img, test_img)
                cv.imshow("Aligned", aligned_img)
                print("Aligned image displayed")

    cam.release()
    cv.destroyAllWindows()


main()
