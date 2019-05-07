import os
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

### TODO: Write data loaders for training, validation, and test sets
## Specify appropriate transforms, and batch_sizes
from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt
import numpy as np

size = 160
cropped_size = 128

normalize = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5],
)

transform = transforms.Compose([
    transforms.Resize(size),
    transforms.CenterCrop(cropped_size),
    transforms.ToTensor()
    #,normalize
])

#data_dir = "./dogImages"
data_dir = "./dogImagesSample"
# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '/train', transform=transform)
test_data = datasets.ImageFolder(data_dir + '/test', transform=transform)
valid_data = datasets.ImageFolder(data_dir + '/valid', transform=transform)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

class_names = [item[4:].replace("_", " ") for item in train_data.classes]
num_classes = len(class_names)
print(class_names)


def show_dogs(images):
    fig, axes = plt.subplots(figsize=(10, 4), ncols=4)
    for ii in range(4):
        ax = axes[ii]
        img = images[ii] / 2 + 0.5  # unnormalize
        ax.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image


data_iter = iter(trainloader)

images, labels = next(data_iter)
show_dogs(images)

images, labels = next(data_iter)
images, labels = next(data_iter)
images, labels = next(data_iter)
show_dogs(images)

images, labels = next(data_iter)
images, labels = next(data_iter)
images, labels = next(data_iter)
show_dogs(images)

plt.show()

loaders_scratch = dict()
loaders_scratch['train'] = trainloader
loaders_scratch['test'] = testloader
loaders_scratch['valid'] = validloader
