## TODO: Specify data loaders
### TODO: Write data loaders for training, validation, and test sets
## Specify appropriate transforms, and batch_sizes
from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt
import cv2

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

size = 128
cropped_size = 128

train_transforms = transforms.Compose([
    transforms.Resize(size),
    transforms.CenterCrop(size),
    transforms.ToTensor()])

test_transforms = transforms.Compose([transforms.Resize(size),
                                      transforms.CenterCrop(size),
                                      transforms.ToTensor()])

data_dir = "./dogImages"
# data_dir = "./dogImagesSample"
# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
valid_data = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

data_iter = iter(trainloader)

# images, labels = next(data_iter)
# fig, axes = plt.subplots(figsize=(10, 4), ncols=4)
# for ii in range(4):
#     ax = axes[ii]
#     ax.imshow(images[ii].numpy().transpose((1, 2, 0)))

# plt.show()

loaders_transfer = dict()
loaders_transfer['train'] = trainloader
loaders_transfer['test'] = testloader
loaders_transfer['valid'] = validloader

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

model_transfer = models.vgg16(pretrained=True)
n_inputs = model_transfer.classifier[6].in_features
# add last linear layer (n_inputs -> 133 dog breed classes)
# new layers automatically have requires_grad = True
last_layer = nn.Linear(n_inputs, 133)
model_transfer.classifier[6] = last_layer
for param in model_transfer.features.parameters():
    param.requires_grad = False

print(model_transfer)

class_names = [item[4:].replace("_", " ") for item in loaders_transfer['train'].dataset.classes]
print(class_names)
use_cuda = False

from PIL import Image


# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    ## TODO: Complete the function.

    return True


def predict_breed_transfer(img_path):
    # load the image and return the predicted breed

    image = Image.open(img_path)

    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop((255, 255)),
        transforms.ToTensor()
    ])

    tensor = transform(image)

    out = model_transfer.forward(tensor.reshape(1, 3, 255, 255))
    if use_cuda:
        out = out.cpu()
    idx_of_max_value = out.detach().numpy().argmax()

    #    plt.imshow(tensor.numpy().transpose((1, 2, 0)))
    #    plt.show()
    #    print(out)
    #    print(idx_of_max_value)
    return class_names[idx_of_max_value]


def run_app(img_path):
    greeting = 'Hi you! Impossible to say whether you are human or dog'
    tell_me_more = ''

    is_dog = dog_detector(img_path)

    if is_dog:
        greeting = 'Hi dog!'
        breed = predict_breed_transfer(img_path)
        tell_me_more = 'You look like a ... {}'.format(breed)
    else:
        is_human = face_detector(img_path)
        if is_human:
            greeting = 'Hi human!'
            breed = predict_breed_transfer(img_path)
            tell_me_more = 'You look like a...\n {}'.format(breed)

    image = Image.open(img_path)
    plt.text(-10, -100, greeting, fontsize=12)

    plt.imshow(image)
    plt.text(-10, image.height + 150, tell_me_more, fontsize=12)
    plt.show()


run_app("./dogImagesSample/valid/001.Affenpinscher/Affenpinscher_00038.jpg")
# breed = predict_breed_transfer("./dogImagesSample/valid/001.Affenpinscher/Affenpinscher_00038.jpg")
# print(breed)
