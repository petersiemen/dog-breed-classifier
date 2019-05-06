import numpy as np
from glob import glob

import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm

dog_files = np.array(glob("dogImages/*/*/*"))
print('There are %d total dog images.' % len(dog_files))

# img = cv2.imread(dog_files[1])
#
# print(img.shape)
# plt.imshow(img)
# plt.show()

# for dog_file in dog_files:
#     img = cv2.imread(dog_file)
#     shape = img.shape
#     sum_height += shape[0]
#     sum_width += shape[1]
#     total += 1


import concurrent.futures

sum_height = 0.
sum_width = 0.
total = 0


def inspect_dog_file(dog_file):
    img = cv2.imread(dog_file)
    shape = img.shape
    height = shape[0]
    width = shape[1]
    return (height, width)


with concurrent.futures.ProcessPoolExecutor() as executor:
    # Process the list of files, but split the work across the process pool to use all CPUs!
    for image_file, (height, width) in zip(dog_files, executor.map(inspect_dog_file, dog_files)):
        # print(f"A thumbnail for {image_file} was saved as {thumbnail_file}")
        sum_height += height
        sum_width += width
        total += 1

print("Average height: {}".format(sum_height / total))
print("Average width: {}".format(sum_width / total))

# There are 8351 total dog images.
# Premature end of JPEG file
# Average height: 529.0449048018202
# Average width: 567.0325709495869


import torch.optim as optim
import torch.nn as nn


### TODO: select loss function
criterion_scratch = nn.CrossEntropyLoss()

### TODO: select optimizer
optimizer_scratch = optim.SGD(model_scratch.parameters(), lr=0.01)
