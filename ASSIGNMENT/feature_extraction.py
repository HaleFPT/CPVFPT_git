import mediapipe as mp
import cv2
import os
import csv

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

def extract_face_features(image):
    # preprocess image (resize, convert to RGB, etc.)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 480))

    # Extract face features
    results = face_mesh.process(image)

    features = []
    # Extract landmarks
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):
                features.append(landmark.x)
                features.append(landmark.y)
                features.append(landmark.z)
    return features






