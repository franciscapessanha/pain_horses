#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 15:57:34 2021

@author: Pessa001
"""

# %%============================================================================
#      IMPORTS AND INITIALIZATIONS
# ==============================================================================

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import pickle
import cv2 as cv
from mpl_toolkits.mplot3d import Axes3D
import glob
from sklearn.model_selection import train_test_split, StratifiedKFold
import shutil
import pandas as pd
from itertools import product
from collections import Counter
import math
from collections import Counter
import copy


DATASET = os.path.join(os.getcwd(), '..', 'dataset')
MODELS =  os.path.join(DATASET, '3D_annotations', 'models')
LANDMARKS =  os.path.join(DATASET, '3D_annotations', 'landmarks')
ANGLES =  os.path.join(DATASET, '3D_annotations', 'angles')

EXCLUDED = os.path.join(os.getcwd(), '..', 'dataset','excluded')

if not os.path.exists(EXCLUDED):
    os.mkdir(EXCLUDED)


data = pickle.load(open(os.path.join(DATASET, 'lms_annotations.pkl'), "rb"))
all_profile = [i for i in data.values if (i[2] == -60 or i[2] == 60) and i[1] == 'horse']
all_tilted = [i for i in data.values if (i[2] == -30 or i[2] == 30) and i[1] == 'horse']
all_frontal = [i for i in data.values if (i[2] == 0) and i[1] == 'horse']


frontal_angles =  pickle.load(open(os.path.join(ANGLES, 'frontal_roll_pitch_yaw.pickle'), "rb"))
tilted_angles =  pickle.load(open(os.path.join(ANGLES, 'tilted_roll_pitch_yaw.pickle'), "rb"))
profile_angles =  pickle.load(open(os.path.join(ANGLES, 'profile_roll_pitch_yaw.pickle'), "rb"))

excluded_frontal = [img[0] for img in all_frontal if img[0].split('/')[-1] not in frontal_angles[:,0]]
excluded_tilted = [img[0] for img in all_tilted if img[0].split('/')[-1] not in tilted_angles[:,0]]
excluded_profile = [img[0] for img in all_profile if img[0].split('/')[-1] not in profile_angles[:,0]]

excluded_images = np.hstack([excluded_frontal, excluded_tilted, excluded_profile])

for img_path in excluded_images:
    shutil.copy2(img_path, os.path.join(EXCLUDED, img_path.split('/')[-1]))