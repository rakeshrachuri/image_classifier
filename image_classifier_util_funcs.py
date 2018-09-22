
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import json
def get_data_transforms():
    
    # Define your transforms for the training, validation, and testing sets
    data_transforms = {'train_transforms': transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]),
                   'test_transforms': transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])]),
                   'valid_transforms': transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
                  }
    return data_transforms

def get_datasets_and_dataloaders(data_dir):
    data_transforms = get_data_transforms()
    #  Load the datasets with ImageFolder
    image_datasets = { 'train_data': datasets.ImageFolder(data_dir + '/train', transform=data_transforms['train_transforms']),
                   'test_data': datasets.ImageFolder(data_dir + '/test', transform=data_transforms['test_transforms']),
                   'valid_data': datasets.ImageFolder(data_dir + '/valid', transform=data_transforms['valid_transforms'])
                 }

    #  Using the image datasets and the trainforms, define the dataloaders
    dataloaders = { 'train_loader': torch.utils.data.DataLoader(image_datasets['train_data'], batch_size=64, shuffle=True),
                'test_loader': torch.utils.data.DataLoader(image_datasets['test_data'], batch_size=32, shuffle=True),
                'valid_loader': torch.utils.data.DataLoader(image_datasets['valid_data'], batch_size=32)
             }
    return image_datasets, dataloaders

def process_image(image_loc):
    
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    img = Image.open(image_loc)
    
    img.thumbnail((256,256))
    width, height = img.size
    print(img.size)
    new_width, new_height = (224,224)
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    img = img.crop((left, top, right, bottom))
    np_image = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    #print('shape before transpose = {}'.format(np_image.shape))
    #print('shape after transpose = {}'.format(np.transpose(np_image).shape))
    print(np_image.shape)
    np_image = ((np_image - mean)/std )   /255
    #np_image = np.transpose(np_image, (2, 0, 1))
    return np.transpose(np_image, (2, 0, 1))