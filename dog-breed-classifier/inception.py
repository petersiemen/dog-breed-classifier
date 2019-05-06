import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Set PIL to be tolerant of image files that are truncated.
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# define VGG16 model
Inception_v3 = models.inception_v3(pretrained=True)

# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move model to GPU if CUDA is available
if use_cuda:
    Inception_v3 = Inception_v3.cuda()

print("use_cuda: {}".format(use_cuda))


def Inception_predict(img_path):
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
        transforms.Resize(360),
        transforms.CenterCrop(299),
        transforms.ToTensor()
    ])

    tensor = transform(image)
    if use_cuda:
        tensor = tensor.cuda()

    reshaped = tensor.reshape(1, 3, 299, 299)
    out = Inception_v3(reshaped)
    if use_cuda:
        out = out.cpu()

    idx_of_max_value = out.detach().numpy().argmax()

    return idx_of_max_value  # predicted class index



prediction = Inception_predict('dogImages/train/001.Affenpinscher/Affenpinscher_00001.jpg')
print(prediction)

#Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

#(Conv2d_1a_3x3): BasicConv2d(
#    (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
#(bn): BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
#)
