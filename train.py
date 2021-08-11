''' Importation des modules - '''
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models

import torch.optim.lr_scheduler as lr_scheduler
from collections import OrderedDict
import json
import PIL
from PIL import Image


import time
import argparse
import utilities_2



args = argparse.ArgumentParser(description='Train.py')

args.add_argument('data_dir', nargs='?', action="store", default="./flowers/", help='dataset folder used for training, validation and testing')
args.add_argument('--gpu', dest="gpu", action="store_true", default="False", help='use gpu for training')
args.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint_2.pth", help='checkpoint for saving the trained model')
args.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001, help='learning rate for the training')
args.add_argument('--epochs', dest="epochs", action="store", type=int, default=8, help='number of epochs for the training')
args.add_argument('--arch', dest="arch", action="store", choices=["vgg19","vgg19"], default="vgg19", type = str, help='sets the architecture of the network')
args.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=512, help='number of nodes in the hidden layer of the classifier')

args = args.parse_args()
data_dir = args.data_dir
save_dir = args.save_dir
lr = args.learning_rate
learning_rate = args.learning_rate
arch = args.arch
hidden_units= args.hidden_units
gpu = args.gpu
epochs = args.epochs

print("data_dir : ", data_dir)
print("gpu :", gpu)
print("arch :", arch)
print("epochs :", epochs)
print("lr : ", lr)
print("hidden_units", hidden_units)
print("save_dir", save_dir)


with open('cat_to_name.json', 'r') as f:
    flower_to_name = json.load(f)
    flower_species = len(flower_to_name)


image_datasets, dataloaders = utilities_2.load_data(data_dir) # load data
model = utilities_2.model_setup(arch, hidden_units, flower_species, gpu) # création du modèle
criterion, optimizer = utilities_2.optimizer_setup(model, lr) # optimisation du modèle


training_loader = dataloaders['train']
validation_loader = dataloaders['valid']
model = utilities_2.train(model, epochs, learning_rate, criterion, optimizer, training_loader, validation_loader)


#utilities.test(dataloaders, model, criterion, gpu)
utilities_2.save_checkpoint('./model_transfer_densenet161.pt', arch, image_datasets, model, hidden_units, flower_species)



#
