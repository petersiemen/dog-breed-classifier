import torchvision.models as models
from torchvision import datasets, transforms
import torch.nn as nn
import torch
import os
import logging
from PIL import Image


def root_dir():  # pragma: no cover
    return os.path.abspath(os.path.dirname(__file__))


class Classifier:
    device = torch.device('cpu')
    model = models.vgg16(pretrained=True)
    n_inputs = model.classifier[6].in_features
    # add last linear layer (n_inputs -> 133 dog breed classes)
    # new layers automatically have requires_grad = True
    last_layer = nn.Linear(n_inputs, 133)
    model.classifier[6] = last_layer
    for param in model.features.parameters():
        param.requires_grad = False

    model_transfer_pt = os.path.join(root_dir(), 'resources/model_transfer.pt')
    model.load_state_dict(torch.load(model_transfer_pt, map_location=device))

    cropped_size = 224
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),

        transforms.RandomResizedCrop(cropped_size),
        transforms.ToTensor(),
        normalize
    ])

    class_names = ['Affenpinscher', 'Afghan hound', 'Airedale terrier', 'Akita', 'Alaskan malamute',
                   'American eskimo dog', 'American foxhound', 'American staffordshire terrier',
                   'American water spaniel', 'Anatolian shepherd dog', 'Australian cattle dog', 'Australian shepherd',
                   'Australian terrier', 'Basenji', 'Basset hound', 'Beagle', 'Bearded collie', 'Beauceron',
                   'Bedlington terrier', 'Belgian malinois', 'Belgian sheepdog', 'Belgian tervuren',
                   'Bernese mountain dog', 'Bichon frise', 'Black and tan coonhound', 'Black russian terrier',
                   'Bloodhound', 'Bluetick coonhound', 'Border collie', 'Border terrier', 'Borzoi', 'Boston terrier',
                   'Bouvier des flandres', 'Boxer', 'Boykin spaniel', 'Briard', 'Brittany', 'Brussels griffon',
                   'Bull terrier', 'Bulldog', 'Bullmastiff', 'Cairn terrier', 'Canaan dog', 'Cane corso',
                   'Cardigan welsh corgi', 'Cavalier king charles spaniel', 'Chesapeake bay retriever', 'Chihuahua',
                   'Chinese crested', 'Chinese shar-pei', 'Chow chow', 'Clumber spaniel', 'Cocker spaniel', 'Collie',
                   'Curly-coated retriever', 'Dachshund', 'Dalmatian', 'Dandie dinmont terrier', 'Doberman pinscher',
                   'Dogue de bordeaux', 'English cocker spaniel', 'English setter', 'English springer spaniel',
                   'English toy spaniel', 'Entlebucher mountain dog', 'Field spaniel', 'Finnish spitz',
                   'Flat-coated retriever', 'French bulldog', 'German pinscher', 'German shepherd dog',
                   'German shorthaired pointer', 'German wirehaired pointer', 'Giant schnauzer',
                   'Glen of imaal terrier', 'Golden retriever', 'Gordon setter', 'Great dane', 'Great pyrenees',
                   'Greater swiss mountain dog', 'Greyhound', 'Havanese', 'Ibizan hound', 'Icelandic sheepdog',
                   'Irish red and white setter', 'Irish setter', 'Irish terrier', 'Irish water spaniel',
                   'Irish wolfhound', 'Italian greyhound', 'Japanese chin', 'Keeshond', 'Kerry blue terrier',
                   'Komondor', 'Kuvasz', 'Labrador retriever', 'Lakeland terrier', 'Leonberger', 'Lhasa apso',
                   'Lowchen', 'Maltese', 'Manchester terrier', 'Mastiff', 'Miniature schnauzer', 'Neapolitan mastiff',
                   'Newfoundland', 'Norfolk terrier', 'Norwegian buhund', 'Norwegian elkhound', 'Norwegian lundehund',
                   'Norwich terrier', 'Nova scotia duck tolling retriever', 'Old english sheepdog', 'Otterhound',
                   'Papillon', 'Parson russell terrier', 'Pekingese', 'Pembroke welsh corgi',
                   'Petit basset griffon vendeen', 'Pharaoh hound', 'Plott', 'Pointer', 'Pomeranian', 'Poodle',
                   'Portuguese water dog', 'Saint bernard', 'Silky terrier', 'Smooth fox terrier', 'Tibetan mastiff',
                   'Welsh springer spaniel', 'Wirehaired pointing griffon', 'Xoloitzcuintli', 'Yorkshire terrier']

    @staticmethod
    def classify(image_file):
        img_path = image_file
        image = Image.open(img_path)

        tensor = Classifier.transform(image)
        out = Classifier.model.forward(tensor.reshape(1, 3, Classifier.cropped_size, Classifier.cropped_size))
        idx_of_max_value = out.detach().numpy().argmax()

        logging.info("classifying {}".format(image_file))
        return Classifier.class_names[idx_of_max_value]
