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





args = argparse.ArgumentParser(description='predict.py')
args.add_argument('--input_img', action="store", type = str, default='flowers/test/28/image_05230.jpg', help='chemin daccès pour les images')
args.add_argument('--checkpoint', nargs='*', action="store",type = str, default='./checkpoint.pth', help='checkpoint utilisé pour charger le modèle entrainé')
args.add_argument('--top_k', dest="top_k", action="store", type=int,  default=5, help='renvoie les catégories les plus probables')
args.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json', help='sse mapping des catégories aux noms réels')
args.add_argument('--gpu', action="store_true", dest="gpu", default="False", help='GPU')
args.add_argument('--arch', dest="arch", action="store", choices=["vgg19","vgg19"], default="vgg19", type = str, help='sets the architecture of the network')


args = args.parse_args()
image_path = args.input_img
top_k = args.top_k
gpu = args.gpu
checkpoint = args.checkpoint
category_names = args.category_names
arch = args.arch

print("image_path : ", image_path)
print("top_k : ", top_k)
print("gpu : ", gpu)
print("checkpoint : ", checkpoint)
print("category_names : ", category_names)


'''
x = image_path.split("/")

with open(category_names, 'r') as f:
    flower_to_name = json.load(f)
flower_species=len(flower_to_name)
target_class=flower_to_name[x[-2]]

idx_to_class = {model.class_to_idx[i]:i for i in model.class_to_idx.keys()}
classes = [flower_to_name[idx_to_class[c]] for c in kclass]
data = {'Predicted Class':classes,'Probablity': value}
dframe = pd.DataFrame(data)


print("Image file:", image_path)
print("Target class:", target_class)
print(dframe.to_string(index = False))
'''





with open('cat_to_name.json', 'r') as f:
    flower_to_name = json.load(f)
    flower_species = len(flower_to_name)


path = 'checkpoint.pth'
image = ("flowers/test/4/image_05658.jpg")


model = utilities_2.load_model(path, flower_species)
model = utilities_2.process_image(image)
#model = utilities_2.imshow(image, ax=None, title=None)




img = 'image_04563.jpg'
path = './flowers/test/40/' + img
model_2 = models.vgg19(pretrained=True)

idx_to_class = {model_2.class_to_idx[i]:i for i in model_2.class_to_idx.keys()}
#classes = [flower_to_name[idx_to_class[c]] for c in kclass]
#data = {'Predicted Class':classes,'Probablity': value}
#dframe = pd.DataFrame(data)



model = utilities_2.predict(path, model_2, topk=5)



#


#
