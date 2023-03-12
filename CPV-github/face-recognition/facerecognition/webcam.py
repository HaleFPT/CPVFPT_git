"""Webcam utilities."""

import cv2

import gfx


KEY_ESC = 27


def capture():
    # vc = cv2.VideoCapture(0)
    vc = cv2.VideoCapture('rtsp://192.168.31.211:8080/h264_pcm.sdp')
    # vc = cv2.VideoCapture('http://192.618.31.211:8080')
    if vc.isOpened():
        while True:
            success, frame = vc.read()
            if success:
                cv2.imshow('Webcam', frame)
                key = cv2.waitKey(10)
                if key == KEY_ESC:
                    break
        cv2.destroyWindow('Webcam')
        return gfx.Image(frame)
    else:
        raise RuntimeError('Failed to open webcam.')


def display():
    # vc = cv2.VideoCapture(0)
    vc = cv2.VideoCapture('rtsp://192.168.1.40:8080/h264_pcm.sdp')
    # vc = cv2.VideoCapture('http://192.618.31.211:8080')
    key = 0
    success = True

    face_detector = gfx.FaceDetector()

    while success and key != KEY_ESC:
        success, frame = vc.read()
        face_detector.show(gfx.Image(frame), wait=True)
        key = cv2.waitKey(10)
        # destroy the window if the user presses 'q'
        if key == ord('q'):
            cv2.destroyWindow('FaceAttend')
            break

