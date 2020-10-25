#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 15:17:09 2020

@author: franciscapessanha
"""


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

DATASET = os.path.join(os.getcwd(),'..', 'dataset')

data = pickle.load(open(os.path.join(DATASET, 'lms_annotations.pkl'), "rb"))

data_frontal = [values for values in data.values if values[2] == 0]
data_tilted = [values for values in data.values if abs(values[2]) == 30]
data_profile = [values for values in data.values if abs(values[2]) == 60]

for data in data_frontal[:10]:
	img = cv.imread(os.path.join(os.getcwd(), '..', data[0]))
	lms = data[-1]

	for pt in lms:
		cv.circle(img,  tuple([int(pt[0]), int(pt[1])]), 4, (255,0,0), -1)

	int_lms = [lms[i] for i in range(len(lms)) if i in [0, 2, 4, 5, 7, 9, *range(10,35), *range(43,46)]]

	for pt in int_lms:
		cv.circle(img,  tuple([int(pt[0]), int(pt[1])]), 4, (255,0,255), -1)

	cv.imwrite('frontal_' + data[0].split('/')[-1], img)

#%%

for data in data_tilted[:10]:
	img = cv.imread(os.path.join(os.getcwd(), '..', data[0]))
	lms = data[-1]

	for pt in lms:
		cv.circle(img,  tuple([int(pt[0]), int(pt[1])]), 4, (255,0,0), -1)

	int_lms = [lms[i] for i in range(len(lms)) if i in [0, 2, 4, 5, 7, 9,  *range(10,33)]]

	for pt in int_lms:
		cv.circle(img,  tuple([int(pt[0]), int(pt[1])]), 4, (255,0,255), -1)

	cv.imwrite('tilted_' + data[0].split('/')[-1], img)

for data in data_profile[:10]:
	img = cv.imread(os.path.join(os.getcwd(), '..', data[0]))
	lms = data[-1]

	for pt in lms:
		cv.circle(img,  tuple([int(pt[0]), int(pt[1])]), 4, (255,0,0), -1)

	int_lms = [lms[i] for i in range(len(lms)) if i in [0, 2, 5, *range(6,10),*range(12,38)]]

	for pt in int_lms:
		cv.circle(img,  tuple([int(pt[0]), int(pt[1])]), 4, (255,0,255), -1)

	cv.imwrite('profile_' + data[0].split('/')[-1], img)


#for label , indexes in [['frontal',[0, 2, 4, 5, 7, 9, *range(10,35), *range(43,46)]], ['tilted', [0, 2, 4, 5, 7, 9,  *range(10,33)]], ['profile', [0, 2, 5, *range(6,10),*range(12,38)]]]:
