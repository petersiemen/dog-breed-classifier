import numpy as np
from glob import glob

import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm

# load filenames for human and dog images
human_files = np.array(glob("lfw/*/*"))
dog_files = np.array(glob("dogImages/*/*/*"))

# print number of images in each dataset
print('There are %d total human images.' % len(human_files))
print('There are %d total dog images.' % len(dog_files))

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[0])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x, y, w, h) in tqdm(faces):
    # add bounding box to color image
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()


# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


human_files_short = human_files[:100]
dog_files_short = dog_files[:100]

detected_humans = [face_detector(human_file) for human_file in tqdm(human_files_short)].count(True)
detected_dogs = [face_detector(dog_file) for dog_file in tqdm(dog_files_short)].count(True)


print('\nTest Accuracy (Human): %2d%% (%2d/%2d)' % (
    100. * detected_humans / len(human_files_short),
    detected_humans, len(human_files_short)))

print('\nTest Accuracy (Dog): %2d%% (%2d/%2d)' % (
    100. * detected_dogs / len(dog_files_short),
    detected_dogs, len(dog_files_short)))


import face_recognition
def dlib_face_detector(img_path):
    image = face_recognition.load_image_file(img_path)
    face_locations = face_recognition.face_locations(image, model="cnn")
    return len(face_locations) > 0

dlib_detected_humans = [dlib_face_detector(human_file) for human_file in tqdm(human_files_short)].count(True)
print('\nTest Accuracy (Human): %2d%% (%2d/%2d)' % (
    100. * dlib_detected_humans / len(human_files_short),
    dlib_detected_humans, len(human_files_short)))
