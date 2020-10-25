#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 16:35:55 2020

@author: franciscapessanha
"""
import pickle
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import pandas as pd

DATASET = os.path.join(os.getcwd(), '..', 'dataset')
ANGLES =  os.path.join(DATASET, '3D_annotations', 'angles')
N_FOLDS = 3

data = pickle.load(open( os.path.join(DATASET, 'lms_annotations.pkl'), "rb" ))
train_set = glob.glob((os.path.join(DATASET, 'cross_val', 'train', '*.jpg')))

columns = ['path', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'yaw', 'pitch', 'roll']

def save_dataframe(all_info, name):
	df_ = pd.DataFrame(columns=columns)
	for angles in all_info:
		index = int(angles[0].split('.')[0])
		img_path = os.path.join(DATASET, 'images', angles[0].replace('png', 'jpg'))
		img = cv.imread(img_path)

		info = data.values[index - 1]
		lms = np.vstack(info[-1])
		lms_x = lms[:,0]
		lms_y = lms[:,1]

		img_h, img_w = img.shape[:2]
		x_min =  max(0,int(min(lms_x)))
		x_max = min(img_w, int(max(lms_x)))

		y_min = max(0, int(min(lms_y)))
		y_max = min(img_h, int(max(lms_y)))

		roll = float(angles[1])
		pitch = float(angles[2])
		yaw = float(angles[3])

		dic = {'path': img_path,
				 'bbox_x_min': x_min,
				 'bbox_y_min': y_min,
				 'bbox_x_max': x_max,
				 'bbox_y_max': y_max,
				 'yaw': yaw,
				 'pitch': pitch,
				 'roll': roll}
		df_.loc[len(df_)] = dic

		df_.to_csv (name + '.csv', index = False, header=True)

for k in range(N_FOLDS):
	all_train = []
	all_val = []

	for label in ['frontal', 'tilted', 'profile']:
		angles_complete_train = np.vstack(np.load(open(os.path.join(ANGLES, '%s_roll_pitch_yaw.pickle' % (label)), 'rb'), allow_pickle = True))
		angles_val = np.vstack(np.load(open(os.path.join(ANGLES, '%s_val_fold_%d_angles.pickle' % (label, k)), 'rb'), allow_pickle = True))
		angles_train = np.vstack([i for i in angles_complete_train if i not in angles_val])

		all_train.append(angles_train)
		all_val.append(angles_val)

	all_train = np.vstack(all_train)
	all_val = np.vstack(all_val)

	save_dataframe(all_train, 'train_k_%d' % k)
	save_dataframe(all_train, 'val_k_%d' % k)
