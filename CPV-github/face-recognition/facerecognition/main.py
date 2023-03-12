import sys
import os

import webcam


def ensure_dir_exists(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def take_training_photos(name, n):
    for i in range(n):
        for face in webcam.capture().faces():
            normalized = face.gray().scale(100, 100)

            face_path = 'C:\\Python Projects\\CPVFPT_git\\CPV-github\\face-recognition\\training_images\\{}'.format(name)
            ensure_dir_exists(face_path)
            normalized.save_to('{}/{}.pgm'.format(face_path, i + 1))

            normalized.show()

def train():
    name = input('Enter your name: ')
    take_training_photos(name, 10)


def main():
    choices = int(input('Enter 1 for train, 2 for demo: '))
    if choices == 1:
        train()
        main()
    elif choices == 2:
        webcam.display()
        main()
    else:
        print('Invalid choice.')
        main()

if __name__ == '__main__':
    main()
