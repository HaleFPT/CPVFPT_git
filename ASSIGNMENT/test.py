import mediapipe as mp
from scipy.spatial.distance import cosine

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2, min_detection_confidence=0.5)

image1 = mp.solutions.face_mesh.FaceMesh.process_image('input_images/Embe/0.jpg')
image2 = mp.solutions.face_mesh.FaceMesh.process_image('input_images/Ha/0.jpg')

landmarks1 = face_mesh.process(image1).multi_face_landmarks[0]
landmarks2 = face_mesh.process(image2).multi_face_landmarks[0]

feature_vector1 = []
feature_vector2 = []

for landmark in landmarks1:
    feature_vector1.append(landmark.x)
    feature_vector1.append(landmark.y)
    feature_vector1.append(landmark.z)

for landmark in landmarks2:
    feature_vector2.append(landmark.x)
    feature_vector2.append(landmark.y)
    feature_vector2.append(landmark.z)


similarity = 1 - cosine(feature_vector1, feature_vector2)

threshold = 0.9

if similarity > threshold:
    print("The faces in the images belong to the same person.")
else:
    print("The faces in the images belong to different people.")
