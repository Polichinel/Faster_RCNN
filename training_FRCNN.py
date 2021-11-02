# Basic python and ML Libraries
import os
import random
import math
import numpy as np
import pandas as pd
# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')

# We will be reading images using OpenCV
import cv2

# xml library for parsing xml files
#from xml.etree import ElementTree as et

from xml.etree import ElementTree, ElementInclude
from collections import Counter

# matplotlib for visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# torchvision libraries
import torch
import torchvision
from torchvision import transforms as torchtrans  
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# these are the helper libraries imported.
from engine import train_one_epoch, evaluate
import utils
import transforms as T

# my own utils
from utils_FRCNN import plot_img_bbox
from utils_FRCNN import train_one_epoch # corrected foir linear_lr error
from utils_FRCNN import MyDataset # the data loader
from utils_FRCNN import get_object_detection_model
from utils_FRCNN import get_transform
from utils_FRCNN import train_test_split
from utils_FRCNN import apply_nms
from utils_FRCNN import torch_to_pil

# for transformations
import albumentations as A
from albumentations.pytorch import ToTensorV2


files_dir = '/home/projects/ku_00017/data/raw/bodies/OD_images_annotated'

dataset_full = MyDataset(files_dir) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
print('length of dataset = ', len(dataset_full), '\n')

# getting the image and target for a test index.  Feel free to change the index.
img, target = dataset_full[78]
print(img.shape, '\n',target)

# creating dir for figures:
figures_path = '/home/projects/ku_00017/people/simpol/scripts/bodies/Faster_RCNN/figures'
os.makedirs(figures_path, exist_ok=True)

# plotting the image with bboxes. Feel free to change the index
img, target = dataset_full[25]
fig_path = '/home/projects/ku_00017/people/simpol/scripts/bodies/Faster_RCNN/figures/plot_initial_eks.jpg'
plot_img_bbox(img, target, fig_path)

# path = os.path.join(files_dir, "images")

# use our dataset and defined transformations
#dataset_full = MyDataset(files_dir) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# dataset_test = FruitImagesDataset(files_dir, 480, 480, transforms= get_transform(train=False))
# so you do not change size... you prop should...

# # split the dataset in train and test set OLD
# torch.manual_seed(1)
# indices = torch.randperm(len(dataset)).tolist()

# # split the dataset in train and test set
# indices = torch.randperm(len(dataset_full)).tolist()
# dataset = torch.utils.data.Subset(dataset_full, indices[:-160])
# dataset_test = torch.utils.data.Subset(dataset_full, indices[-160:])

# # split the dataset in train and test set
dataset, dataset_test = train_test_split(dataset_full)

print(f'number of images in trainset: {len(dataset)}')
print(f'number of images in testset: {len(dataset_test)}')


# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=8, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

# to train on gpu if selected.
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = len(dataset_full.classes) #minus backgorund?

# get the model using our helper function
model = get_object_detection_model(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# training for 10 epochs
# num_epochs = 10
num_epochs = 1000


for epoch in range(num_epochs):
    # training for one epoch
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

# pick one image from the test set
img, target = dataset[0]

# put the model in evaluation mode
model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])[0]
    
print(f'img shape: {img.shape}')
print(f'img shape (squeeze): {img.squeeze().shape}')
print(f'img shape (squeeze, axis moved): {torch.moveaxis(img.squeeze(), 0, -1).shape}')


#plot EXPECTED OUTPUT
fig_path_EO = '/home/projects/ku_00017/people/simpol/scripts/bodies/Faster_RCNN/figures/plot_EO.jpg'
plot_img_bbox(torch_to_pil(img), target, fig_path_EO)
print('EXPECTED OUTPUT plotted')

#plot MODEL OUTPUT
fig_path_MO = '/home/projects/ku_00017/people/simpol/scripts/bodies/Faster_RCNN/figures/plot_MO.jpg'
plot_img_bbox(torch_to_pil(img), prediction, fig_path_MO)
print('MODEL OUTPUT plotted')

# NON-MAX-Surpresssion
nms_prediction = apply_nms(prediction, iou_thresh=0.2)
fig_path_NMS = '/home/projects/ku_00017/people/simpol/scripts/bodies/Faster_RCNN/figures/plot_NMS.jpg'
plot_img_bbox(torch_to_pil(img), nms_prediction, fig_path_NMS)
print('NMS APPLIED MODEL OUTPUT plottet')

# lets try the test set:
img, target = dataset_test[0]

# put the model in evaluation mode
model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])[0]
    
fig_path_EO_test = '/home/projects/ku_00017/people/simpol/scripts/bodies/Faster_RCNN/figures/plot_EO_test.jpg'
plot_img_bbox(torch_to_pil(img), target, fig_path_EO_test)
print('EXPECTED (test) OUTPUT plotted\n')

fig_path_NMS_test = '/home/projects/ku_00017/people/simpol/scripts/bodies/Faster_RCNN/figures/plot_NMS_test.jpg'
nms_prediction = apply_nms(prediction, iou_thresh=0.01)
plot_img_bbox(torch_to_pil(img), nms_prediction, fig_path_NMS_test)
print('MODEL OUTPUT (test) plotted\n')

model_path = '/home/projects/ku_00017/people/simpol/scripts/bodies/Faster_RCNN/my_model_weights.pth'
torch.save(model.state_dict(), model_path)
print('Model weights saved!')
