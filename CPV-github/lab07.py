import numpy as np
import cv2 as cv
import os
from PIL import Image
import tkinter as tk

mov = 'images/Tiktok video.mov'
cap = cv.VideoCapture(mov)
# 393x852
cap.set(3, 852)
cap.set(4, 393)

def capture_image():
    print("capturing")
    # crop mov to 393x852

    face_detector = cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')

    # For each person, enter one numeric face id
    face_id = input('enter user id:  ')
    name = input('enter user name:  ')

    #check if the csv file exists
    if not os.path.exists('StudentDetails.csv'):
        with open('StudentDetails.csv', 'w') as f:
            f.write('id,name,\n')

    # check if the id is already in the csv file
    with open('StudentDetails.csv', 'r') as f:
        for line in f:
            if face_id in line:
                print("ID already exists")
                return

    # save name and id in separate columns in csv file
    with open('StudentDetails.csv', 'a') as f:
        f.write(f'{face_id},{name},\n')

    # check if the dataset folder exists and create it if it doesn't
    if not os.path.exists('dataset'):
        os.mkdir('dataset')

    count = 0

    print("\nInitializing face capture. Look the camera and wait ...")

    while True:
        ret, img = cap.read()
        if ret:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            # wait for 5 seconds to capture image after pressing c
            for (x, y, w, h) in faces:
                cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                count += 1
                # Save the captured image into the datasets folder
                cv.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])
                cv.imshow('image', img)
            k = cv.waitKey(100) & 0xff  # Press 'ESC' for exiting video
            if k == 27:
                break
            elif count >= 30:  # Take 30 face sample and stop video
                break
    cap.release()
    cv.destroyAllWindows()
    print("Image captured")

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
    recognizer.write('trained/trainer.yml')  # recognizer.save() worked on Mac, but not on Pi

    # Print the numer of faces trained and end program
    print("\n{0} faces trained. Exiting Program".format(len(np.unique(ids))))
    print("Training complete")


def recognizer():
    print("recognizing")
    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.read('trained/trainer.yml')
    cascadePath = "haarcascade_frontalface_alt2.xml"
    faceCascade = cv.CascadeClassifier(cascadePath)

    font = cv.FONT_HERSHEY_SIMPLEX

    # iniciate id counter
    id = []

    # check if the dataset folder exists and create it if it doesn't
    if not os.path.exists('dataset'):
        os.makedirs('dataset')

    # check if the dataset have images
    if len(os.listdir('dataset')) == 0:
        print("No images in dataset folder")
        return

    # extract name from dataset folder
    for i in os.listdir('dataset'):
        id.append(i.split('.')[1])

    # names related to ids (StudentDetails.csv)
    names = []
    with open('StudentDetails.csv', 'r') as f:
        for line in f:
            names.append(line.split(',')[1])

    # Define min window size to be recognized as a face


    while True:
        ret, img = cap.read()
        if ret:
            minW = 0.1 * img.shape[1]
            minH = 0.1 * img.shape[0]
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
        else:
            break
    cap.release()
    cv.destroyAllWindows()

def delete_images():
    for i in os.listdir('dataset'):
        os.remove('dataset/' + i)
    print("Images deleted")

def delete_trained():
    os.remove('trained/trainer.yml')
    print("Trained deleted")

def delete_csv():
    os.remove('StudentDetails.csv')
    print("CSV deleted")

def delete_all():
    delete_images()
    delete_trained()
    delete_csv()
    print("All deleted")

def main():
    controler = tk.Tk()
    controler.title("Controler")
    controler.geometry("150x200")
    capture = tk.Button(controler, text="Capture", command=lambda: capture_image())
    train = tk.Button(controler, text="Train", command=lambda: training())
    recognize = tk.Button(controler, text="Recognize", command=lambda: recognizer())
    capture.pack()
    train.pack()
    recognize.pack()
    controler.mainloop()


if __name__ == '__main__':
    delete_all()
    # main()