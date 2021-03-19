#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 17:53:41 2021

@author: Pessa001
"""

# %%============================================================================
#                          IMPORTS AND INITIALIZATIONS
# ==============================================================================

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import pickle
import cv2 as cv
from mpl_toolkits.mplot3d import Axes3D
import glob
from sklearn.model_selection import train_test_split
import shutil

BIG_SIDE = 600

DATASET = os.path.join(os.getcwd(),'..','dataset')
ANGLES =  os.path.join(DATASET, '3D_annotations', 'angles')

#LMS_SYSTEM = 'complete' # complete or absolute

data = pickle.load(open(os.path.join(DATASET, 'lms_annotations.pkl'), "rb"))

IMAGES_WITH_LMS = os.path.join(os.getcwd(), '..', 'dataset','images_with_lms')

if not os.path.exists(IMAGES_WITH_LMS):
    os.mkdir(IMAGES_WITH_LMS)


def resize_img(img, lms):
    init_h, init_w = np.shape(img)[:2]
    ratio = init_h/init_w
    if ratio < 1:
        width = BIG_SIDE
        height = BIG_SIDE * ratio
    else:
        height = BIG_SIDE
        width = BIG_SIDE / ratio

    img_resize = cv.resize(img, (int(width), int(height)))
    lms_resize = []
    for pt in lms:
        new_pt = (pt[0]  * width/init_w, pt[1] * height/init_h)
        #new_pt = ((pt[0] - x_min), (pt[1] - y_min))
        lms_resize.append(new_pt)

    lms_resize = np.vstack(lms_resize)
    return img_resize, lms_resize

for folder in ['horse', 'donkey','']:
    path = os.path.join(IMAGES_WITH_LMS,folder)
    if not os.path.exists(path):
        os.mkdir(path)

    for folder in ['frontal', 'tilted','profile']:
        sub_path = os.path.join(path, folder)
        if not os.path.exists(sub_path):
            os.mkdir(sub_path)

for animal in data.values:
    img = cv.imread(animal[0])
    img_name = animal[0].split('/')[-1]
    lms = animal[-1]
    pose = animal[2]
    species = animal[1]

    img, lms = resize_img(img, lms)
    if abs(pose) == 0: #frontal
        pose = 'frontal'
        relative_indexes = [0, 2, 4, 5, 7, 9, *range(10,35), *range(43,46)]
        outline_lms = [lms[i] for i in range(len(lms)) if i in relative_indexes]
        abs_lms = [lms[i] for i in range(len(lms)) if i not in relative_indexes]

    elif abs(pose) == 30: #tilted
        pose = 'tilted'
        relative_indexes = [0, 2, 4, 5, 7, 9,  *range(10,33)]
        outline_lms = [lms[i] for i in range(len(lms)) if i in relative_indexes]
        abs_lms = [lms[i] for i in range(len(lms)) if i not in relative_indexes]

    elif abs(pose) == 60: #profile
        pose = 'profile'
        relative_indexes = [0, 2, 5, *range(6,10),*range(12,38)]
        outline_lms = [lms[i] for i in range(len(lms)) if i in relative_indexes]
        abs_lms = [lms[i] for i in range(len(lms)) if i not in relative_indexes]

    for (x,y) in outline_lms:
        r = 4
        cv.circle(img, (int(x), int(y)), r, (0,255,0), thickness=-1, lineType=cv.LINE_AA)

    for (x,y) in abs_lms:
        r = 4
        cv.circle(img, (int(x), int(y)), r, (0,0,255), thickness=-1, lineType=cv.LINE_AA)

    cv.imwrite(os.path.join(IMAGES_WITH_LMS, species, pose, img_name), img)

