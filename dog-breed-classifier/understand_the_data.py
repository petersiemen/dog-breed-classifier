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

sum_height = 0.
sum_width = 0.
total = 0
for dog_file in dog_files:
    img = cv2.imread(dog_file)
    shape = img.shape
    sum_height += shape[0]
    sum_width += shape[1]
    total += 1
    # print(img.shape)
    # print("sum_width: {}".format(sum_width))
    # print("sum_height: {}".format(sum_height))
    # plt.imshow(img)
    # plt.show()
    # break

print("Average height: {}".format(sum_height / total))
print("Average width: {}".format(sum_width / total))

# There are 8351 total dog images.
# Premature end of JPEG file
# Average height: 529.0449048018202
# Average width: 567.0325709495869



