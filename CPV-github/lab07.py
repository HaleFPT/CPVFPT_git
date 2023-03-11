import numpy as np
import cv2 as cv
from PIL import Image

mov = 'images/Tiktok video.mov'
feed = cv.imread('images/Feed.png')
navBar = cv.imread('images/Nav bar.png')
sideBar = cv.imread('images/Sidebar.png')

def button_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        print("Button clicked at ({}, {})".format(x, y))
        # call your desired function here, e.g. capture_image()
        capture_image(img)

def capture_image(img):
    face_detector = cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')

    # For each person, enter one numeric face id
    face_id = input('enter user id:  ')
    name = input('enter user name:  ')

    # save name and id in separate columns in csv file
    with open('StudentDetails.csv', 'a') as f:
        f.write(f'{face_id},{name},\n')
    print("\nInitializing face capture. Look the camera and wait ...")
    # Initialize individual sampling face count
    count = 0
    while (True):
        img = cv.flip(img, -1)  # flip video image vertically
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1

            # Save the captured image into the datasets folder
            cv.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

        if count >= 30:  # Take 30 face sample and stop video
            break

    # Do a bit of cleanup
    print("\nExiting Program and cleanup stuff")


def training():
    # Path for face image database
    path = 'dataset'

    recognizer = cv.face.LBPHFaceRecognizer_create()
    detector = cv.CascadeClassifier("haarcascade_frontalface_alt2.xml")

    # function to get the images and label data
    def getImagesAndLabels(path):

        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []

        for imagePath in imagePaths:

            PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
            img_numpy = np.array(PIL_img, 'uint8')

            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)

        return faceSamples, ids

    print("\nTraining faces. It will take a few seconds. Wait ...")
    faces, ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))
    # Save the model into trainer/trainer.yml
    recognizer.write('trainer/trainer.yml')  # recognizer.save() worked on Mac, but not on Pi

    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))


def recognizer(img):
    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = "haarcascade_frontalface_alt2.xml"
    faceCascade = cv.CascadeClassifier(cascadePath);

    font = cv.FONT_HERSHEY_SIMPLEX

    # iniciate id counter
    id = []
    # extract name from dataset folder
    for i in os.listdir('dataset'):
        id.append(i.split('.')[1])

    # names related to ids (StudentDetails.csv)
    names = []
    with open('StudentDetails.csv', 'r') as f:
        for line in f:
            names.append(line.split(',')[1])

    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        # img = cv.flip(img, -1)  # Flip vertically
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (92, 208, 220), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            # Check if confidence is less then 50% identify it as unknown
            if (confidence < 50):
                id = names[id - 1]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv.putText(img, str(confidence), (x + 8, y + h - 8), font, 1, (255, 255, 0), 1)

        cv.imshow('camera', img)

        k = cv.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break


def main():
    # crop mov to 393x852
    cap = cv.VideoCapture(mov)
    # 393x852
    cap.set(3, 852)
    cap.set(4, 393)
    # create black window for menu control using opencv
    control = np.zeros((852, 393, 3), np.uint8)

    cv.setMouseCallback('control', button_callback)
    while True:
        ret, frame = cap.read()
        if ret:
            # crop frame
            frame = frame[84:852, 0:393]
            feed[0:852-84, 0:393] = frame

            cv.imshow('frame', feed)

            # add a button to the window
            button_text = "Capture Image"
            button_position = (50, 50)
            button_size = (200, 50)
            cv.rectangle(control, button_position,
                          (button_position[0] + button_size[0], button_position[1] + button_size[1]), (0, 255, 0),
                          thickness=-1)
            cv.putText(control, button_text, (button_position[0] + 10, button_position[1] + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv.imshow('control', control)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()