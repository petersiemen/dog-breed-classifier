from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.transforms.functional as TF
import torch
import torch.nn as nn

# Set PIL to be tolerant of image files that are truncated.
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
import matplotlib.pyplot as plt

# Load the pretrained model from pytorch
vgg16 = models.vgg16(pretrained=True)


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

    # conv = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # conv.forward(tensor.reshape(1,3,255,255))
    # print(conv)

    # plt.imshow(tensor.numpy().squeeze(), cmap='Greys_r');

    # plt.imshow(tensor.numpy().transpose((1, 2, 0)))
    # plt.show()
    # print(vgg16)
    out = vgg16(tensor.reshape(1, 3, 255, 255))
    idx_of_max_value = out.detach().numpy().argmax()

    return idx_of_max_value  # predicted class index


prediction = VGG16_predict('dogImages/train/001.Affenpinscher/Affenpinscher_00001.jpg')
print(prediction)
