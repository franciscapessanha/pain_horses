#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:32:43 2020

@author: franciscapessanha
"""

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import pickle
import cv2 as cv
import trimesh
import glob
from data_handling.create_pts_lms import crop_image, resize_img, create_pts
from data_augmentation.utils import *
#from load_obj import *
#from transformations import *
#import math
from utils import *
#import random

DATASET = os.path.join(os.getcwd(), 'dataset')
COLORS =  os.path.join(DATASET, '3D_annotations', 'colors')
SHAPES =  os.path.join(DATASET, '3D_annotations', 'shapes')
ANGLES =  os.path.join(DATASET, '3D_annotations', 'angles')
MODELS =  os.path.join(DATASET, '3D_annotations', 'models')


EX_FOLDER = os.path.join(os.getcwd(), 'examples')
N_FOLDS = 3

ABS_POSE = os.path.join(os.getcwd(), 'dataset','abs_pose')
if not os.path.exists(ABS_POSE):
	os.mkdir(ABS_POSE)

for folder in ['frontal', 'tilted','profile']:
	path = os.path.join(ABS_POSE,folder)
	if not os.path.exists(path):
		os.mkdir(path)

	for k in range(N_FOLDS):
		sub_path = os.path.join(path,'aug_data_%d' % k)
		if not os.path.exists(sub_path):
			os.mkdir(sub_path)


TEMP = os.path.join(os.getcwd(), 'temp')

if not os.path.exists(TEMP):
	os.mkdir(TEMP)
def read_pts(filename):
    return np.loadtxt(filename, comments=("version:", "n_points:", "{", "}"))

for k in range(N_FOLDS):
	for pose, number, ratio in [['profile', 30, 600/330]]:
		images = glob.glob(os.path.join(ABS_POSE, pose, 'data_aug_1.5', '*.png'))
		for img_path in images:
			img = cv.imread(img_path)

			lms = read_pts(img_path.replace('png', 'pts'))
			img, lms = flip_image(img, lms, number)
			#if len(lms) > 29:
				#img, lms = flip_image(img, lms, number)
				#img, lms = crop_image(img, lms, number)
				#img_resize, lms_resize = resize_img(img, lms, ratio)


				#create_pts(lms, img_path.split('.')[0] + '.pts')
			cv.imwrite(img_path, img)

			#int_lms = np.vstack([lms[i] for i in pickle.load(open(os.path.join(MODELS, pose + '_indexes.pickle'), 'rb'))])
			create_pts(lms, img_path.replace('png', 'pts'))

