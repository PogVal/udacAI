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



'''
import time
import argparse
import utilities_2
'''







''' Importation des images, leurs transformations et la création de 3 catégories : entrainement, validation, test '''
def load_data(data_dir):
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Définir les transformations pour les données servant à la formation, au test et à la validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(35),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),

        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),

        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Les 3 catégories d'images
    image_datasets = {
        'train' : datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid' : datasets.ImageFolder(test_dir, transform=data_transforms['valid']),
        'test' : datasets.ImageFolder(valid_dir, transform=data_transforms['test'])
    }

    dataloaders = {
        'train' : torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid' : torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32, shuffle=True),
        'test' : torch.utils.data.DataLoader(image_datasets['test'], batch_size=32, shuffle=False)
    }

    #print("data_transforms", data_transforms)
    return image_datasets, dataloaders



    #print("Données chargée OK")
    #print("image_datasets : ", image_datasets)
    #print("dataloaders: ", dataloaders)



# ------------------------------------------ Tester "load_data"
#data_dir = 'flowers'
#load_data(data_dir)







''' Function MODEL_SETUP '''

arch = 'vgg19'



def model_setup(arch, hidden_units, flower_species, gpu):

    '''
    with open('cat_to_name.json', 'r') as f:
        flower_to_name = json.load(f)
        #print(flower_to_name)
    '''


    ''' Définir un nouveau réseau feed-forward non entraîné (pour les dernières couches)
    comme classificateur en utilisant comme fonction d'activation Dropout et ReLU. '''
    # CHOIX D'UN NETWORK PRÉ-ENTRAINÉ (Transfer learning)

    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    else:
        print("{} Pas de modèle valide".format(arch))


    # pas de mise à jour des poids du modèle pré-entraîné mais seulement ceux du classificateur (False)
    for param in model.parameters():
        param.requires_grad=False

    model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 1024)), # 1er couche
                          ('drop', nn.Dropout(p=0.5)), # activation function
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(1024, 102)), # couche de sortie
                          ('output', nn.LogSoftmax(dim=1)) # loss function
                          ]))

    # GPU
    use_cuda = torch.cuda.is_available() and gpu
    if use_cuda:
        model = model.cuda()

    #print("model_setup OK")
    #print("model : ", model) #---> model VGG

    return model



# ------------------------------------------ Tester "Model_setup"
#model_setup("vgg19", 20, "pink primrose", False)










''' Entrainement du modèle : utilise les couches du classificateur en utilisant la rétro-propagation
(et le réseau pré-entraîné pour obtenir les caractéristiques/features)'''

def optimizer_setup(model,lr):
    criterion = nn.NLLLoss()  # nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)

    # Decay LR by a factor of 0.1 every 4 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    #print("optimizer_setup() ------")
    #print("criterion : ", criterion)
    #print("optimizer : ", optimizer)

    return criterion, optimizer


# ------------------------------------------ Tester "Optimizer"
#print("test optimizer")
#model = models.vgg19(pretrained=True)
#optimizer_setup(model, 0.01)



# --------------------------  Function TRAIN(n_epochs, loaders, model, optimizer, criterion, gpu, save_path)  -----------------------------
''' Entrainement du modèle : utilise les couches du classificateur en utilisant la rétro-propagation (et le réseau pré-entraîné pour obtenir les caractéristiques/features)'''

def train(model, epochs, learning_rate, criterion, optimizer, training_loader, validation_loader):

    model.train() # Train mode
    print_every = 5
    steps = 0
    use_gpu = False

    # si disponibilité du GPU
    if torch.cuda.is_available():
        use_gpu = True
        model.cuda()
    else:
        model.cpu()

    # Itération à travers chaque passage d'entrainement - Dépends du nombre d'Epoch
    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in iter(training_loader):
            steps += 1

            if use_gpu: # si disponibilité du GPU
                inputs = Variable(inputs.float().cuda())
                labels = Variable(labels.long().cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels)

            # Forward and backward passes
            optimizer.zero_grad() # zero's out the gradient, otherwise will keep adding

            output = model.forward(inputs) # Forward propogation
            loss = criterion(output, labels) # Calculates loss

            loss.backward() # Calcul du Gradient
            optimizer.step() # Mise à jour des Weights selon le Gradient et le Learning rate
            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss, accuracy = validate(model, criterion, validation_loader) # voir function plus bas (6)

                print("Epoch: {}/{} ".format(epoch+1, epochs),
                        "TRAINING LOSS: {:.3f} ".format(running_loss/print_every),
                        "VALIDATION LOSS: {:.3f} ".format(validation_loss),
                        "VALIDATION ACCURACY: {:.3f}".format(accuracy))

    return model


''' Validation du modèle (nécessaire par la fonction d'entrainement)- '''

def validate(model, criterion, data_loader):
    model.eval() # évaluation/validation du modèle
    exactitude = 0 # initialisation des variables
    perte = 0

    for inputs, labels in iter(data_loader):
        if torch.cuda.is_available():  # si disponibilité du GPU
            inputs = Variable(inputs.float().cuda())
            labels = Variable(labels.long().cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)

        output = model.forward(inputs) # sortie

        perte += criterion(output, labels).item() # incrémentation de la perte

        tenseur = torch.exp(output).data # Retourne un nouveau tenseur avec l'exponentielle des éléments du tenseur d'entrée
        idem = (labels.data == tenseur.max(1)[1])# vérifie si les valeurs correspondent entre elles

        exactitude += idem.type_as(torch.FloatTensor()).mean() # moyenne

    return perte / len(data_loader), exactitude / len(data_loader) # renvoie le résultat : perte et exactitude du modèle




# ------------------------------------------ Tester "Train"
'''
model = models.vgg19(pretrained=True)
data_dir = 'flowers'
training_loader = torch.utils.data.DataLoader(datasets.ImageFolder(data_dir + '/train'), batch_size=64, shuffle=True)
validation_loader = torch.utils.data.DataLoader(datasets.ImageFolder(data_dir + '/valid'), batch_size=64, shuffle=True)
epochs = 5
learning_rate = 0.01
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

train(model, epochs, learning_rate, criterion, optimizer, training_loader, validation_loader)
'''



# ------------------------------------------ Tester "SaveCke"

'''
Enregistrement du modèle pour le réutiliser plus tard
'''

def save_checkpoint(path, arch, image_datasets, model, hidden_units, flower_species):
    model.class_to_idx = image_datasets['train'].class_to_idx
    torch.save({'structure' : arch,
            'state_dict': model.state_dict(),
            'hidden_units': hidden_units,
            'flower_species': flower_species,
            'class_to_idx': model.class_to_idx},
            path)
    print("save_checkpoint OK")





''' --------------------- Télécharger le modèle enregistré IIIII --------------------- '''
def load_model(path, flower_species):
    checkpoint = torch.load('checkpoint.pth')
    #structure = save_checkpoint['arch']


    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    else:
        print("{} Pas de modèle valide".format(arch))


    for param in model.parameters():
        param.requires_grad=False

    '''
    model.classifier= nn.Sequential(
                    nn.Linear(model.classifier.in_features,checkpoint['hidden_layer']),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(checkpoint['hidden_layer'], flower_species)
                    )

    '''
    learning_rate = checkpoint['learning_rate']
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    #optimizer.load_state_dict(checkpoint['optimizer'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    print("load model Ok")
    return model




def process_image(image):
    img = Image.open(image)
    img.load()
    img = img.resize((256,256))
    value = 0.5*(256-224)
    img = img.crop((value,value,256-value,256-value))
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    img = img.transpose((2, 0, 1))

    print("process image OK")
    return img


''' --------------------- Affichage de l'image test -> Imshow for Tensor --------------------- '''

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    image = image.numpy().transpose((1, 2, 0)) # la couche couleur comme première dimension pour PyTorch et la troisième pour matplotlib
    mean = np.array([0.485, 0.456, 0.406]) # moyenne
    std = np.array([0.229, 0.224, 0.225]) # standart déviation
    image = std * image + mean
    image = np.clip(image, 0, 1) # L'image doit être coupée entre 0 et 1, sinon elle ressemble à du bruit lorsqu'elle est affichée
    ax.imshow(image)

    print("imshow OK")
    return ax


''' --------------------- Prédiction de la CLASS- avec/sans l'utilisation du GPU --------------------- '''


def predict(path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    '''
    cuda = torch.cuda.is_available()
    if cuda:
        # Move model parameters to the GPU
        model.cuda()
        print("Number of GPUs:", torch.cuda.device_count())
        print("Device name:", torch.cuda.get_device_name(torch.cuda.device_count()-1))
    else:
        model.cpu()
        print("We go for CPU")
    '''
    # turn off dropout
    model.eval()

    # The image
    image = process_image(path)

    # tranfer to tensor
    image = torch.from_numpy(np.array([image])).float()

    # The image becomes the input
    image = Variable(image)

    '''
    if cuda:
        image = image.cuda()
    '''
    output = model.forward(image)

    probabilities = torch.exp(output).data
    prob = torch.topk(probabilities, topk)[0].tolist()[0] # probabilities
    index = torch.topk(probabilities, topk)[1].tolist()[0] # index



    ind = []
    for i in range(len(model.class_to_idx.items())):
        ind.append(list(model.class_to_idx.items())[i][0])


    # transfer index to label

    label = []
    for i in range(5):
        label.append(ind[index[i]])


    print("predict OK")
    return prob, label




#
