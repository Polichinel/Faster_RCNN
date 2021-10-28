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

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Create data loader class:
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms = None, n_obs = 50):
        self.root = root
        self.transforms = transforms
        self.n_obs = n_obs

        # the selection need to happen here
        # self.classes = [''] + self.__get_classes__() # list of classes accroding to n_obs, see __get_classes__
        # self.classes = [_] + self.__get_classes__() # list of classes accroding to n_obs, see __get_classes__
        self.classes = ['background'] + self.__get_classes__() # list of classes accroding to n_obs, see __get_classes__
        self.classes_int = np.arange(0,len(self.classes)) # from 1 since no background '0'
        self.boxes = self.__get_boxes__() # list of xml files (box info) to n_obs, see __get_classes__
        self.imgs = [f"{i.split('.')[0]}.jpg" for i in self.boxes] # list of images - only take images with box info! and > n_obs
             
    def __get_classes__(self):
        """Creates a list of classes with >= n_obs observations"""
        n_obs = self.n_obs
        path = self.root

        obj_name = []
        #classes = []

        # Get all objects that have been annotated
        for filename in os.listdir(path):
            if filename.split('.')[1] == 'xml':
                box_path = os.path.join(path, filename)

                tree = ElementTree.parse(box_path)
                lst_obj = tree.findall('object')

                for j in lst_obj:
                    obj_name.append(j.find('name').text)


        # now, only keep the objects w/ >= n_obs observations
        # c = Counter(obj_name)

        # for i in c.items():

        #  #   this might be the issue! Changing the int here does not change it in the xml file... 
        #     if i[1] >= n_obs:
        #         classes.append(i[0])

        #     classes.append(i[0])    # whar does this do now?     
        
        classes = sorted(set(obj_name))

        return(classes)

    def __get_boxes__(self):
        """Make sure you only get images with valid boxes frrom the classes list - see __get_classes__"""

        path = self.root

        boxes = []
        # Get all objects that have been annotated
        for filename in os.listdir(path):
            if filename.split('.')[1] == 'xml':
                box_path = os.path.join(path, filename)

                tree = ElementTree.parse(box_path)
                lst_obj = tree.findall('object')

                # If there is one or more objects from the classes list, save the box filename
                if len(set([j.find('name').text for j in lst_obj]) & set(self.classes)) > 0:
                    boxes.append(filename)

        # Sort and return the boxes
        # boxes = sorted(boxes) # is this what is fucking you up?!?!?!?!?!??!?!?!??!?!?
        return(boxes)

    def __getitem__(self, idx):
        # dict to convert classes into classes_int
        class_to_int = dict(zip(self.classes,self.classes_int)) # is it here???!?!?!   

        
        img_path = os.path.join(self.root, self.imgs[idx])
        box_path = os.path.join(self.root, self.boxes[idx])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize img 800x800 --------------------------------------------
        target_size = 800

        y_orig_size = img.shape[0] # the original y shape
        x_orig_size = img.shape[1] # the original x shape
        y_scale = target_size/y_orig_size # scale factor for boxes
        x_scale = target_size/x_orig_size # scale factor for boxes

        img = cv2.resize(img, (target_size, target_size))
        # ----------------------------------------------------------------

        img = np.moveaxis(img, -1, 0) # move channels in front so h,w,c -> c,h,w
        img = img / 255.0 # norm ot range 0-1. Might move out..
        img = torch.Tensor(img)

        # Open xml path 
        tree = ElementTree.parse(box_path)

        lst_obj = tree.findall('object')

        obj_name = []
        obj_ids = []
        boxes = []

        for i in lst_obj:
        # here you need to ignore classes w/ n > n_obs

            obj_name_str = i.find('name').text
            if obj_name_str in self.classes:

                obj_name.append(obj_name_str) # get the actual class name
                obj_ids.append(class_to_int[i.find('name').text]) # get the int associated with the class name
                lst_box = i.findall('bndbox')

                for j in lst_box:

                    xmin = float(j.find('xmin').text) * x_scale # scale factor to fit resized image
                    xmax = float(j.find('xmax').text) * x_scale
                    ymin = float(j.find('ymin').text) * y_scale
                    ymax = float(j.find('ymax').text) * y_scale
                    boxes.append([xmin, ymin, xmax, ymax])
            else:
                pass

        num_objs = len(obj_ids) # number of objects

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(obj_ids, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes 
        target["labels"] = labels
        target["image_id"] = image_id 
        target["area"] = area
        target["iscrowd"] = iscrowd 

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs) # right now you do not differentiate between annotated images and not annotated images... 


    def target_classes(self):
        t_inst_classes = dict(zip(self.classes_int,self.classes)) # just a int to string dict
        return(t_inst_classes)

    def coco_classes(self):
        inst_classes = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'] # a "ordered" list of the coco categories. should remove perhaps?
        return(inst_classes) 


# Create functions
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger

# Function to visualize bounding boxes in the image - save the image in the figures folder
def plot_img_bbox(img, target, fig_path):

    # Remove old plot if it exists:
    if os.path.exists(fig_path):
        os.remove(fig_path)

    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    fig, a = plt.subplots(1,1)
    fig.set_size_inches(5,5)

    # if the funciton torch_to_pil have not been apllied
    if type(img) == torch.Tensor:
        img = torch.moveaxis(img, 0, -1)
    elif type(img) == np.ndarray:
        img = np.moveaxis(img, 0, -1)
    else: 
        pass

    a.imshow(img)
    for box in (target['boxes']):
        # specify cpu just in case.
        x, y, width, height  = box[0].cpu(), box[1].cpu(), box[2].cpu()-box[0].cpu(), box[3].cpu()-box[1].cpu()
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth = 2,
                                 edgecolor = 'r',
                                 facecolor = 'none')

        # Draw the bounding box on top of the image
        a.add_patch(rect)
    plt.savefig(fig_path, bbox_inches = "tight")

# TRY RETINANET - e.i. focal loss!!! 
def get_object_detection_model(num_classes, retrain_all_param = True):

    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True) # you should be able to change pretrained from here..
    
    # Retrain all parameters if retrain_all_param == True
    for param in model.parameters():
        param.requires_grad = retrain_all_param

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    # Per default new layers have requires_grad = True, so if retrain_all_param = False, this layer will still train 

    return model

def get_object_detection_model_retina(num_classes, retrain_all_param = True):

    # load a model pre-trained pre-trained on COCO
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True) # you should be able to change pretrained from here..
    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)

    # Retrain all parameters if retrain_all_param == True
    for param in model.parameters():
        param.requires_grad = retrain_all_param

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    # Per default new layers have requires_grad = True, so if retrain_all_param = False, this layer will still train 

    return model

    # Send train=True fro training transforms and False for val/test transforms
def get_transform(train):
    
    if train:
        return A.Compose([
                            A.HorizontalFlip(0.5),
                     # ToTensorV2 converts image to pytorch tensor without div by 255
                            ToTensorV2(p=1.0) 
                        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    else:
        return A.Compose([
                            ToTensorV2(p=1.0)
                        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

# the function takes the original prediction and the iou threshold.

def train_test_split(dataset_full, train_ratio = 0.85):
    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset_full)).tolist()

    n_images = len(indices)

    indices_train = indices[int(-n_images*train_ratio):]
    indices_test = indices[:int(-n_images*train_ratio)]

    dataset_train = torch.utils.data.Subset(dataset_full, indices_train)
    dataset_test = torch.utils.data.Subset(dataset_full, indices_test)

    return(dataset_train, dataset_test)

def apply_nms(orig_prediction, iou_thresh=0.3):
    
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]
    
    return final_prediction

# function to convert a torchtensor back to PIL image
def torch_to_pil(img):
    img = img.squeeze()
    return torchtrans.ToPILImage()(img).convert('RGB')

