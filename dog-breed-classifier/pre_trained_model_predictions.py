from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.transforms.functional as TF
import torch
import torch.nn as nn
import numpy as np

use_cuda = False

# Set PIL to be tolerant of image files that are truncated.
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
import matplotlib.pyplot as plt

# Load the pretrained model from pytorch
VGG16 = models.vgg16(pretrained=True)


def VGG16_predict(img_path):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to
    predicted ImageNet class for image at specified path

    Args:
        img_path: path to an image

    Returns:
        Index corresponding to VGG-16 model's prediction
    '''

    ## TODO: Complete the function.
    ## Load and pre-process an image from the given img_path
    ## Return the *index* of the predicted class for that image
    image = Image.open(img_path)
    plt.imshow(image)

    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop((255, 255)),
        transforms.ToTensor()
    ])

    tensor = transform(image)
    if use_cuda:
        tensor = tensor.cuda()

    output = VGG16(tensor.reshape(1, 3, 255, 255))

    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy()) if not use_cuda else np.squeeze(preds_tensor.cpu().numpy())

    return preds


prediction = VGG16_predict('dogImages/train/001.Affenpinscher/Affenpinscher_00001.jpg')
print(prediction)

import numpy as np
from glob import glob

import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm

# load filenames for human and dog images
human_files = np.array(glob("lfw/*/*"))
dog_files = np.array(glob("dogImages/*/*/*"))

human_files_short = human_files[:100]
dog_files_short = dog_files[:100]


### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    ## TODO: Complete the function.
    predicted_class_index = VGG16_predict(img_path)
    is_dog = True if predicted_class_index >= 151 and predicted_class_index <= 268 else False
    return is_dog  # true/false


detected_dogs_in_humans = [dog_detector(human_file) for human_file in tqdm(human_files_short)].count(True)
detected_dogs_in_dogs = [dog_detector(dog_file) for dog_file in tqdm(dog_files_short)].count(True)


print('\nTest Accuracy Dog Classifier (Finding Humans): %2d%% (%2d/%2d)' % (
    100. * detected_dogs_in_humans / len(human_files_short),
    detected_dogs_in_humans, len(human_files_short)))

print('\nTest Accuracy Dog Classifier (Finding Dogs): %2d%% (%2d/%2d)' % (
    100. * detected_dogs_in_dogs / len(dog_files_short),
    detected_dogs_in_dogs, len(dog_files_short)))
