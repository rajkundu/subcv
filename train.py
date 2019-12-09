"""
Training script for PyTorch Darknet model.

e.g. python train.py --cfg cfg/yolov3-tiny-1xclass.cfg --weights yolov3-tiny.weights --datacfg data/obj.data

Output prediction vector is [batch, centre_x, centre_y, box_height, box_width, mask_confidence, class_confidence]
"""

import torch
from torchvision import transforms
import os
import copy
import argparse
from darknet import Darknet, parse_cfg
from util import *
from data_aug.data_aug import Sequence, YoloResizeTransform, Normalize
from preprocess import *
import numpy as np
import cv2
import pickle as pkl
import matplotlib.pyplot as plt
from bbox import bbox_iou, corner_to_center, center_to_corner
import pickle 
from customloader import CustomDataset
import torch.optim as optim
import torch.autograd.gradcheck
from tensorboardX import SummaryWriter
import sys
import datetime
from tqdm import tqdm
import shutil

# Folder to save checkpoints to
SAVE_FOLDER = datetime.datetime.now().strftime("%B-%d-%Y-%I:%M%p")
if os.path.exists(os.path.join(os.getcwd(), 'runs', SAVE_FOLDER)):
    shutil.rmtree(os.path.join(os.getcwd(), 'runs', SAVE_FOLDER))
os.makedirs(os.path.join(os.getcwd(), 'runs', SAVE_FOLDER))

# For tensorboard
writer = SummaryWriter()
random.seed(0)

# Choose backend device for tensor operations - GPU or CPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print("Using device \"%s\"" % device)

def arg_parse():
    """
    Parse arguements to the detect module

    """
    parser = argparse.ArgumentParser(description='YOLO v3 Training Module')
    parser.add_argument("--cfg", dest='cfgfile', help="Config file", type=str, default=os.path.join("cfg","yolov3.cfg"))
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile", type=str, default="yolov3.weights")
    parser.add_argument("--datacfg", dest="datafile",  help="cfg file containing the configuration for the dataset", type=str, default=os.path.join("data","classes.data"))
    parser.add_argument("--lr", dest="lr", type=float, default=1.0)
    parser.add_argument("--rho", dest="rho", type=float, default=0.95)
    parser.add_argument("--eps", dest="eps", type=float, default=1e-5)
    parser.add_argument("--epochs", dest="epochs", type=int, default=5)
    parser.add_argument("--unfreeze", dest="unfreeze", type=int, default=4, help="Last number of layers to unfreeze for training")
    parser.add_argument("--resume", nargs="?", type=str, const=os.path.join("runs","latest.pth"), default="")
    return parser.parse_args()

#Parse arguments
args = arg_parse()

#Load the model
model = Darknet(args.cfgfile, train=True)
model.load_weights(args.weightsfile)

# Unfreeze all but this number of layers at the beginning
layers_length = len(list(model.parameters()))

# "unfreeze" refers to the last number of layers to tune (allow gradients to be tracked - backprop)
stop_layer = layers_length - (args.unfreeze * 2) # Freeze up to this layer (open up more than first phase)

# Load the config file
net_options =  model.net_info
# Parse the config file
batch = net_options['batch']
angle = net_options['angle']    #The angle with which you want to rotate images as a part of augmentation

'''
# For RandomHSV() augmentation (see data_aug.py)
saturation = int(float(net_options['saturation'])*255)    #saturation related augmentation
exposure = int(float(net_options['exposure'])*255)
hue = int(float(net_options['hue'])*179)
'''

# scales = net_options['scales']
num_classes = net_options['classes']
bs = net_options['batch']
# Assume h == w
inp_dim = net_options['height']

inp_dim = int(inp_dim)
num_classes = int(num_classes)
bs = int(bs)

def load_checkpoint(checkpoint_fpath, model, optimizer):
    # Load the state dicts from file
    checkpoint = torch.load(checkpoint_fpath)
    # Load for model
    model.load_state_dict(checkpoint['state_dict'])

    # Unfreeze model & have to re-instantiate optimizer
    unfreeze_layers(model, checkpoint['stop_layer'], False)
    optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, rho=args.rho, eps=args.eps)

    # Load for optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])

    # Convert to CUDA if necessary
    if(torch.cuda.is_available()):
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
    # Convert to CPU if necessary
    else:
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cpu()

    return model, optimizer

def print_param_grad(model):
    for name, param in model.named_parameters():
        print(name +':\t'+str(param.requires_grad))
def freeze_layers(model, stop_layer, offset, printParams=True):
    """Utility to stop tracking gradients in earlier layers of
    NN for transfer learning"""
    cntr = offset
    for name, param in model.named_parameters():
        if cntr < stop_layer or name.find('batch_norm') >= 0:
            param.requires_grad = False
        else:
            param.requires_grad = True
        
        if(printParams):
            print(name +':\t'+str(param.requires_grad))

        cntr+=1
    return model
def unfreeze_layers(model, stop_layer, printParams=True):
    cntr = 0
    for name, param in model.named_parameters():
        if cntr < stop_layer:
            param.requires_grad = False
        else:
            if name.find('batch_norm') < 0:
                param.requires_grad = True
            else:
                param.requires_grad = False

        if(printParams):
            print(name +':\t'+str(param.requires_grad))

        cntr+=1
    return model

def norm(x):
    """Normalize 1D tensor to unit norm"""
    mu = x.mean()
    std = x.std()
    y = (x - mu)/std
    return y

def YOLO_loss(ground_truth, output):
    """Function to calculate loss based on predictions
    and ground-truth labels"""
    total_loss = 0
    
    #get the objectness loss
    loss_inds = torch.nonzero(ground_truth[:,:,-4] > -1)
    objectness_pred = output[loss_inds[:,0],loss_inds[:,1],4]
    target = ground_truth[loss_inds[:,0],loss_inds[:,1],4]
    objectness_loss = torch.nn.MSELoss(reduction='sum')(objectness_pred, target)
    #Only objectness loss is counted for all boxes
    object_box_inds = torch.nonzero(ground_truth[:,:,4] > 0).view(-1, 2)

    try:
        gt_ob = ground_truth[object_box_inds[:,0], object_box_inds[:,1]]
    except IndexError:
        return None
    
    pred_ob = output[object_box_inds[:,0], object_box_inds[:,1]]
    
    #get centre x and centre y 
    centre_x_loss = torch.nn.MSELoss(reduction='sum')(pred_ob[:,0], gt_ob[:,0])
    centre_y_loss = torch.nn.MSELoss(reduction='sum')(pred_ob[:,1], gt_ob[:,1])

    total_loss += centre_x_loss 
    total_loss += centre_y_loss 
    
    #get w,h loss
    w_loss = torch.nn.MSELoss(reduction='sum')(pred_ob[:,2], gt_ob[:,2])
    h_loss = torch.nn.MSELoss(reduction='sum')(pred_ob[:,3], gt_ob[:,3])
    
    total_loss += w_loss 
    total_loss += h_loss 

    #class_loss 
    cls_labels = np.zeros((gt_ob.shape[0], num_classes))
    for i in range(gt_ob.shape[0]):
        cls_labels[i,int(gt_ob[i,5]-1)] = 1
    cls_labels = torch.from_numpy(cls_labels).to(device)
    cls_loss = 0    

    for c_n in range(num_classes):
        targ_labels = pred_ob[:,5 + c_n].view(-1,1)
        targ_labels = targ_labels.repeat(1,2)
        targ_labels[:,0] = 1 - targ_labels[:,0]
        cls_loss += torch.nn.CrossEntropyLoss(reduction='sum')(targ_labels, cls_labels[:,c_n].long())

    total_loss += cls_loss

    # print('-------------------------------')
    # print('class loss ', cls_loss)
    # print('width loss ', w_loss)
    # print('height loss ', h_loss)
    # print('center x loss ', centre_x_loss)
    # print('center y loss ', centre_y_loss)
    # print('-------------------------------')

    return total_loss

### DATA ###

# Overloading custom data transforms from customloader (may add more here)
# custom_transforms = Sequence([RandomHSV(hue=hue, saturation=saturation, brightness=exposure), 
#     YoloResizeTransform(inp_dim), Normalize()])
custom_transforms = Sequence([YoloResizeTransform(inp_dim), Normalize()])

# Data instance and loader
data = CustomDataset(root="data", num_classes=num_classes, 
                     ann_file=os.path.join("data","train.txt"),
                     cfg_file=args.cfgfile,
                     det_transforms=custom_transforms)
print('Batch size ', bs)
data_loader = DataLoader(data, 
                         batch_size=bs,
                         shuffle=True,
                         collate_fn=data.collate_fn)

iterations = len(data)//bs
print('Size of data / batch size (iterations) = {}'.format(iterations))

### TRAIN MODEL ###

# Use CUDA device if available and set to train
model.to(device)
model.train()

if(len(args.resume) > 0):
    optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, rho=args.rho, eps=args.eps)
    if(not os.path.exists(args.resume)):
        print("Checkpoint save file not found: " + args.resume)
        exit()
    model, optimizer = load_checkpoint(args.resume, model, optimizer)

# Freeze layers according to user specification
freeze_layers(model, layers_length - args.unfreeze, 1) # Freeze up until this layer

# Adadelta optimizer
optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, rho=args.rho, eps=args.eps)

itern = 0
total_loss = 0

#Checkpoint every 2 epochs
cpFreq = 2

# Begin
for epoch in range(args.epochs):
    print('Beginning epoch {0}/{1}:'.format(epoch + 1, args.epochs))

    for image, ground_truth in tqdm(data_loader):

        if len(ground_truth) == 0:
            continue

        image = image.to(device)
        ground_truth = ground_truth.to(device)

        # Clear gradients from optimizer for next iteration
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            output = model(image)

        print('...{}'.format(itern), end='')

        if (torch.isnan(ground_truth).any()):
            print("Nans in Ground_truth")
            assert False
            
        if (torch.isnan(output).any()):
            print("Nans in Output")
            assert False
            
        if (ground_truth == float("inf")).any() or (ground_truth == float("-inf")).any():
            print("Inf in ground truth")
            assert False
            
        if (output == float("inf")).any() or (output == float("-inf")).any():
            print("Inf in output")
            assert False

        loss  = YOLO_loss(ground_truth, output)

        if loss:
            total_loss += loss
            loss.backward()
            optimizer.step()

        itern += 1

    print()
    print('Learning Rate:\t', optimizer.param_groups[0]["lr"])
    print("Loss:\t\t{0:.4f}\n".format(float(total_loss)/iterations))
    writer.add_scalar("Loss/vanilla", float(total_loss)/iterations, epoch)

    # Checkpoint
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'stop_layer': stop_layer
    }
    torch.save(checkpoint, os.path.join('runs', 'latest.pth'))
    if epoch % cpFreq == 0 or epoch == args.epochs - 1:
        torch.save(checkpoint, os.path.join('runs', SAVE_FOLDER, 'epoch{0}-bs{1}-loss{2:.4f}-fine.pth'.format(epoch+args.epochs, bs, float(total_loss)/iterations)))

    # Update LR if needed by scheduler
    #scheduler.step()
    total_loss = 0 # reset

### DATA FOR FINE-TUNING ###

# Reset data loader
# Data instance with transforms (augmentations) and PyTorch loader
data = CustomDataset(root="data", num_classes=num_classes, 
                     ann_file=os.path.join("data","train.txt"), 
                     cfg_file=args.cfgfile,
                     det_transforms=custom_transforms)
data_loader = DataLoader(data, batch_size=bs,
                         shuffle=True,
                         collate_fn=data.collate_fn)

### FINE TUNE MODEL ON MORE LAYERS ###

# Open up the whole network
model = unfreeze_layers(model, stop_layer)

# Re-initialize optimizer to include unfrozen layers
optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, rho=args.rho, eps=args.eps)

# For final loss measurement
save_loss = 0

# Begin
for epoch in range(args.epochs):
    print('Beginning epoch {0}/{1}:'.format(epoch + 1, args.epochs))

    for image, ground_truth in tqdm(data_loader):
        if len(ground_truth) == 0:
            continue

        # # Track gradients in backprop
        image = image.to(device)
        ground_truth = ground_truth.to(device)
        
        # Clear gradients from optimizer for next iteration
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            output = model(image)

        print('...{}'.format(itern), end='')

        if (torch.isnan(ground_truth).any()):
            print("Nans in Ground_truth")
            assert False
            
        if (torch.isnan(output).any()):
            print("Nans in Output")
            assert False
            
        if (ground_truth == float("inf")).any() or (ground_truth == float("-inf")).any():
            print("Inf in ground truth")
            assert False
            
        if (output == float("inf")).any() or (output == float("-inf")).any():
            print("Inf in output")
            assert False

        loss  = YOLO_loss(ground_truth, output)
        
        if loss:            
            total_loss += loss
            loss.backward()
            optimizer.step()
 
        itern += 1

    print()
    print('Learning rate:\t', optimizer.param_groups[0]["lr"])
    print("Loss:\t\t{0:.4f}\n".format(float(total_loss)/iterations))
    writer.add_scalar("Loss/vanilla", float(total_loss)/iterations, epoch)
    
    # Checkpoint
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'stop_layer': stop_layer
    }
    torch.save(checkpoint, os.path.join('runs', 'latest.pth'))
    if epoch % cpFreq == 0 or epoch == args.epochs - 1:
        torch.save(checkpoint, os.path.join('runs', SAVE_FOLDER, 'epoch{0}-bs{1}-loss{2:.4f}-fine.pth'.format(epoch+args.epochs, bs, float(total_loss)/iterations)))

    #scheduler.step()
    save_loss = total_loss
    total_loss = 0 # reset

writer.close()
