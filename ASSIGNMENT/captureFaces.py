import cv2
import mediapipe as mp
import os

# Initialize mediapipe Face Mesh
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

if not os.path.exists("input_images"):
    os.makedirs("input_images")

if not os.path.exists("output_images"):
    os.makedirs("output_images")


def take_face():
    cv2.namedWindow("Face")
    vc = cv2.VideoCapture('rtsp://192.168.31.211:8080/h264_pcm.sdp')

    name = input("Enter your name: ")
    if not os.path.exists("C:\\Python Projects\\CPVFPT_git\\ASSIGNMENT\\input_images\\" + name):
        os.makedirs("C:\\Python Projects\\CPVFPT_git\\ASSIGNMENT\\input_images\\" + name)

    count = 0
    while vc.isOpened():
        ret, frame = vc.read()
        if ret:
            # detect face
            with mp_face_mesh.FaceMesh(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as face_mesh:
                results = face_mesh.process(frame)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACE_CONNECTIONS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing.DrawingSpec(
                                color=(80, 110, 10), thickness=1, circle_radius=1))
                        # save image
                        cv2.imwrite("C:\\Python Projects\\CPVFPT_git\\ASSIGNMENT\\input_images\\" + name + "\\" + str(count) + ".jpg", frame)
                        count += 1
                        if count == 10:
                            break
            cv2.imshow("Face", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

def main():
    take_face()

if __name__ == "__main__":
    main()


