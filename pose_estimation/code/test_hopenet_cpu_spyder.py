import sys, os, argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from scipy.stats import pearsonr

import datasets, hopenet, utils

def CCC(t, p):
    astd = np.std(t)
    bstd = np.std(p)
    am = np.mean(t)
    bm = np.mean(p)
    cc = pearsonr(t, p)[0]
    cc = np.nan_to_num(cc)
    o = (2 * cc * astd * bstd) / (pow(astd, 2) + pow(bstd, 2) + pow(am - bm, 2))
    return abs(round(o, 4))


def SAGR(t, p):
    if len(t) == 0 or len(p) == 0:
        print('Empty list for SAGR')
        return 1.0
    if len(t) != len(p):
        print('Length mismatch for SAGR')
        return 0.0
    o = 0
    for i in range(len(t)):
        o += (t[i] > 0) == (p[i] > 0)
    o /= len(t)
    return round(o, 4)


def PCC(t, p):
    cc = pearsonr(t, p)[0]
    cc = np.nan_to_num(cc)
    return abs(round(cc, 4))


def MAE(t, p):
    return round(np.mean(np.abs(t - p)).numpy(), 4)


def evaluate(gt, val):
    print('     Yaw        Pitch      Roll       Ave')
    print('MAE ',
          str(MAE(gt[:, 0], val[:, 0])).ljust(10),
          str(MAE(gt[:, 1], val[:, 1])).ljust(10),
          str(MAE(gt[:, 2], val[:, 2])).ljust(10),
          str(MAE(gt.ravel(), val.ravel())).ljust(10))
    print('PCC ',
          str(PCC(gt[:, 0], val[:, 0])).ljust(10),
          str(PCC(gt[:, 1], val[:, 1])).ljust(10),
          str(PCC(gt[:, 2], val[:, 2])).ljust(10),
          str(PCC(gt.ravel(), val.ravel())).ljust(10))
    print('SAGR',
          str(SAGR(gt[:, 0], val[:, 0])).ljust(10),
          str(SAGR(gt[:, 1], val[:, 1])).ljust(10),
          str(SAGR(gt[:, 2], val[:, 2])).ljust(10),
          str(SAGR(gt.ravel(), val.ravel())).ljust(10))



filename_list = 'test.csv'
data_dir = ''
model_name = ''

cudnn.enabled = True
snapshot_path = os.path.join(os.getcwd(), 'output', 'snapshots', model_name)

model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

print('Loading snapshot.')
saved_state_dict = torch.load(snapshot_path, map_location='cpu')
model.load_state_dict(saved_state_dict)

print('Loading data.')

transformations = transforms.Compose([transforms.Scale(224),
transforms.CenterCrop(224), transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

pose_dataset = datasets.Equines(data_dir, filename_list, transformations)


print('Ready to test network.')

# Test the Model
model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
total = 0

idx_tensor = [idx for idx in range(66)]
idx_tensor = torch.FloatTensor(idx_tensor)

yaw_error = .0
pitch_error = .0
roll_error = .0

l1loss = torch.nn.L1Loss(size_average=False)

gt = []
pred = []


for i, (images, labels, cont_labels, name) in enumerate(test_loader):
    images = Variable(images)
    total += cont_labels.size(0)

    label_yaw = cont_labels[:,0].float()
    label_pitch = cont_labels[:,1].float()
    label_roll = cont_labels[:,2].float()

    gt.append([label_yaw, label_pitch, label_roll])

    yaw, pitch, roll = model(images)

    # Binned predictions
    _, yaw_bpred = torch.max(yaw.data, 1)
    _, pitch_bpred = torch.max(pitch.data, 1)
    _, roll_bpred = torch.max(roll.data, 1)

    # Continuous predictions
    yaw_predicted = utils.softmax_temperature(yaw.data, 1)
    pitch_predicted = utils.softmax_temperature(pitch.data, 1)
    roll_predicted = utils.softmax_temperature(roll.data, 1)


    yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 99
    pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 99
    roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu() * 3 - 99

    pred.append([yaw_predicted, pitch_predicted, roll_predicted])

    # Mean absolute error
    yaw_error += torch.sum(torch.abs(yaw_predicted - label_yaw))
    pitch_error += torch.sum(torch.abs(pitch_predicted - label_pitch))
    roll_error += torch.sum(torch.abs(roll_predicted - label_roll))


evaluate(np.vstack(gt), np.vstack(pred))