# Package import
#from __future__ import logging.info_function, division

'''
import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns 
'''

import math
import numpy as np
import time
import os
import copy
import pandas as pd
import sys

import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F

from torch.autograd import Variable

import config_train


#plt.ion()   # interactive mode

# Set the directories for the data and the CSV files that contain ids/labels

'''
Please note: 

Folders under the directory: dir_images (showed below) have been set up before data input, which means 
there should be two folders named as:
1. 'train' which is randomly selected from all train data (we randomly selected 80%) for training process
2. 'val' which is randomly selected from all train data (we randomly selected 20%) for validation

Every folder above should has saved images in correct subfolders which are labeled as 'Belize' and 'Honduras'.
'''

# create logger
logger = logging.getLogger('IGAM Teacher Train Log')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

dir_images  = './country'

def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

# Define train model
def train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    #predict = []
    #input_grad = []
    for epoch in range(num_epochs):

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.cuda()
                labels = labels.cuda()

                if phase == 'train':
                    inputs = inputs.repeat(1,3,1,1)
                
                #inputs = to_var(inputs, requires_grad=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    #if phase == 'train':
                        #outputs = outputs.logits
                    _, preds = torch.max(outputs, 1)
                    
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        #input_grad.append(inputs.grad.data.cpu().numpy())
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
            
            if (epoch + 1) % 10 ==0 or epoch == 0:
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
                logging.info('-' * 10)
                logging.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                #logging.info()

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict()) 

    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logging.info('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    torch.save(best_model_wts, "./model_ct_grey.pt")
    
    return model

def train(flip_rate, brightness, rd_crop, batch_size, lr, momentum, decay_bd, gamma, num_epochs):
	data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((100,150)),
        transforms.CenterCrop((70,150)),
        transforms.RandomHorizontalFlip(flip_rate),
        transforms.Grayscale(1),
        torchvision.transforms.ColorJitter(brightness=brightness
                                           #, contrast=0.1, saturation=0.1, hue=0.1
                                          ),
        #transforms.RandomRotation(90),
        transforms.RandomCrop(rd_crop),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize([0.485], [0.229])
    ]),
    'val': transforms.Compose([
        transforms.Resize((100,150)),
        #transforms.CenterCrop((100,150)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    }

	data_dir = dir_images
	image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
	                                          data_transforms[x])
	                  for x in ['train', 'val']}
	train_dataloaders = torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size,
	                                             shuffle=True, 
	                                              num_workers=4)

	val_dataloaders = torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size,
	                                             shuffle=False, 
	                                              num_workers=4)

	dataloaders = {'train':train_dataloaders, 'val':val_dataloaders}

	dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
	class_names = image_datasets['train'].classes

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# Original train model is pretrained ResNet34
	#model_ft = nn.DataParallel(models.resnet34(pretrained=True)).cuda()
	model_ft = models.resnet34(pretrained=True).cuda()
	#model_ft = models.inception_v3(pretrained=True)

	#Set up linear layer
	#num_ftrs = model_ft.fc.in_features
	#model_ft.fc = nn.Linear(num_ftrs, 2)

	#model_ft = model_ft.cuda()

	# Loss function
	criterion = nn.CrossEntropyLoss()

	#O ptimizer Stochastic Gradient Decent
	optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=momentum)

	# Decay LR by a factor of 0.1 every 7 epochs
	exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, decay_bd, gamma=gamma)

	model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs, dataloaders=dataloaders, dataset_sizes=dataset_sizes)

def main():
    logging.basicConfig(filename='train_ct_grey.log', level=logging.INFO)
    logging.info('Started')

    args = config_train.get_args()
    args_dict = vars(args)
    train(**args_dict)

    logging.info('Finished')

if __name__ == '__main__':
    main()





