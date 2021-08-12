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

import pandas as pd # -----
import time
import argparse
import utilities_2

import os, random
import tensorflow as tf




args = argparse.ArgumentParser(description='predict.py')
args.add_argument('--input_img', action="store", type = str, default='flowers/test/28/image_05230.jpg', help='url pour les images')
args.add_argument('--checkpoint', nargs='*', action="store", type = str, default='./checkpoint.pth', help='checkpoint > charger le modèle entrainé')
args.add_argument('--top_k', dest="top_k", action="store", type=int,  default=5, help='catégories les plus probables')
args.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json', help='mapping des catégories')
args.add_argument('--gpu', action="store_true", dest="gpu", default="False", help='GPU')
args.add_argument('--arch', dest="arch", action="store", choices=["densenet121","vgg19"], default="vgg19", type = str, help='type architecture pour network')


args = args.parse_args()
image_path = args.input_img
top_k = args.top_k
gpu = args.gpu
checkpoint = args.checkpoint
category_names = args.category_names
arch = args.arch

print("url images : ", image_path)
print("top_k : ", top_k)
print("gpu : ", gpu)
print("checkpoint : ", checkpoint)
print("nom catégories : ", category_names)
print("architecture : ", arch)



# ---------- F > charger le modèle -----------
with open('cat_to_name.json', 'r') as f:
    flower_to_name = json.load(f)

flower_species = len(flower_to_name)
path = 'checkpoint.pth'  
    
model = utilities_2.load_model(path, flower_species)        
print("modèle chargé", model)



# ---------- F > traitement des images -----------
image_path = ("flowers/test/4/image_05658.jpg")
model = utilities_2.process_image(image_path) # -> conversion

#model_imshow = utilities_2.imshow(image, ax=None, title=None)





# ---------- F > prédictions des images -----------

img = random.choice(os.listdir('./flowers/test/40/'))
img_path = './flowers/test/40/' + img

image = utilities_2.process_image(img_path) # --> renvoie un Tensor
#print('image_____________', image)

model = utilities_2.load_model(path, flower_species)
#print('model_____________', model)



path = './flowers/test/40/' + img

prob, classes = utilities_2.predict(img_path, model)


print('#######################################################################')
for i in range(5):
    print('Flower Class : ', flower_to_name[classes[i]], "// Probability : ", prob[i])
    print('-----------------------------------------------------')

print('#######################################################################')





#
