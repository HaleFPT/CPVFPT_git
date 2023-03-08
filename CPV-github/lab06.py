import pathlib as pl

import cv2

imgdir = pl.Path('panorama')


def captureImg():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('frame')

    for i in range(50):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.imwrite(str(imgdir / f'panorama{i}.jpg'), frame)
        cv2.waitKey(1000)

    cap.release()
    cv2.destroyAllWindows()


# panorama stiching
def stitch_images():
    images = []
    # load images
    for i in range(len(list(imgdir.glob('*.jpg')))):
        img = cv2.imread(str(imgdir / f'panorama{i}.jpg'))
        if i == 1:
            images = [img]
        else:
            images.append(img)

    # create stitcher
    stitcher = cv2.Stitcher.create()

    # stitch images
    status, result = stitcher.stitch(images)

    # display result
    if status == cv2.Stitcher_OK:
        cv2.imshow('Stitched image', result)
        cv2.waitKey()
        cv2.destroyAllWindows()
    else:
        print('Stitching failed')


if __name__ == '__main__':
    # captureImg()
    stitch_images()
