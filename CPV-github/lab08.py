import numpy as np
import cv2
import os
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import svm

# Set the path to the dataset folder
dataset_path = 'processed image'

# Load the images from the dataset folder
images = []
names = []
for image_path in os.listdir(dataset_path):
    image = cv2.imread(os.path.join(dataset_path, image_path), cv2.IMREAD_GRAYSCALE)
    images.append(image)
    names.append(image_path.split('.')[0])

names = np.array(names)

def processImg():
    # noramlize the images
    images_normalized = []
    for image in images:
        img = cv2.equalizeHist(image)
        img = cv2.resize(img, (256, 256))
        img = cv2.GaussianBlur(img, (5, 5), 0)
        images_normalized.append(img)

    # flatten the images
    images_flatten = []
    for image in images_normalized:
        img = image.flatten()
        images_flatten.append(img)

    # convert the images to numpy array
    images_flatten = np.array(images_flatten)


    return images_flatten

# split the dataset into training and testing
images_flatten = processImg()
X_train, X_test, y_train, y_test = train_test_split(images_flatten, names, test_size=0.2, random_state=42)

# compute mean image
mean_image = np.mean(X_train, axis=0)

# compute the covariance matrix
cov_matrix = np.cov(X_train.T)

# compute the eigenvalues and eigenvectors of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# sort the eigenvalues and eigenvectors in descending order
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]

#A common rule of thumb is to retain enough principal components to account for 90% or more of the variance in the data. This can be determined by looking at the cumulative sum of the eigenvalues and selecting the number of components that reach the desired threshold.
eigenvalues_sum = np.sum(eigenvalues)
eigenvalues_sum_ratio = eigenvalues / eigenvalues_sum
eigenvalues_sum_ratio_cumsum = np.cumsum(eigenvalues_sum_ratio)
print(eigenvalues_sum_ratio_cumsum)










