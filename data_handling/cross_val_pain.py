#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:17:14 2020

@author: franciscapessanha
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
from sklearn.model_selection import train_test_split, StratifiedKFold
import shutil
import pandas as pd
from itertools import product
from collections import Counter

PLOTS = os.path.join(os.getcwd(), 'plots')
if not os.path.exists(PLOTS):
	os.mkdir(PLOTS)


DATASET = os.path.join(os.getcwd(), '..', 'dataset')
ANGLES =  os.path.join(DATASET, '3D_annotations', 'angles')

CROSS_VAL = os.path.join(os.getcwd(), '..', 'dataset','cross_val_pain')

if not os.path.exists(CROSS_VAL):
	os.mkdir(CROSS_VAL)

for folder in ['frontal', 'tilted','profile']:
	path = os.path.join(CROSS_VAL,folder)
	if not os.path.exists(path):
		os.mkdir(path)

	for sub_folder in ['train', 'test']:
		sub_path = os.path.join(path,sub_folder)
		if not os.path.exists(sub_path):
			os.mkdir(sub_path)

data = pickle.load(open(os.path.join(DATASET, 'lms_annotations.pkl'), "rb"))

pain = pd.read_excel(os.path.join(DATASET, 'pain_annotations.xlsx'), index_col=0)
# %%============================================================================
#                               	AUXILIAR FUNCTIONS
# ==============================================================================

def closer_bin(pt, array):
	k = []
	for i in array:
		k.append(abs(pt - i))

	#print('k index: ', k.index(min(k)))
	return k.index(min(k))

def get_y(yaws, bins, counts):
	y = []
	for yaw in yaws:
		for i in range(len(bins) - 1):
			if yaw >= bins[i] and yaw < bins[i + 1]:
				y.append(i)
				break

	single_classes = np.where(counts == 1)[0]
	null_classes = np.where(counts == 0)[0]

	non_single_classes = [i for i in range(len(counts)) if i not in [*null_classes,*single_classes]]
	updated_y = []
	for label in y:
		if label in single_classes:
			updated_y.append(non_single_classes[closer_bin(label, non_single_classes)])
			print('replaced class: ', non_single_classes[closer_bin(label, non_single_classes)])
		else:
			updated_y.append(label)

	return updated_y

def get_train_test(angles, label):
	x = angles[:, 0]
	photonumbers = [int(i.split('.')[0]) for i in x]
	x_pain = [pain.values[i][1:7] for i in photonumbers]
	combinations = np.asarray(list(product([-1, 0, 1, 2], repeat = 6)))
	y_pain = np.vstack([np.where((combinations == pain).all(axis = 1))[0] for pain in x_pain])
	labels = np.unique(y_pain)
	#print('lenght labels: ', len(labels))
	y_pain = np.vstack([np.where((labels == pain[0]))[0] for pain in y_pain])
	#angles = np.vstack([[fix_angles(float(r),float(p),float(y))] for r,p, y in angles[:,1:]])
	yaws = angles[:, -1].astype(np.float)
	rolls = angles[:, -3].astype(np.float)
	pitchs = angles[:, -2].astype(np.float)

	plt.figure()
	yaw_counts, yaw_bins, _ = plt.hist(yaws, bins=36, range = [-90, 90])
	title = 'Yaw distribution - %s pose' % label
	plt.title(title)
	plt.xlabel('angle (degrees)')
	plt.ylabel('number of occurrences')
	plt.savefig(os.path.join(PLOTS, title))

	plt.figure()
	rolls_counts, rolls_bins, _ = plt.hist(rolls, bins=36, range = [-90, 90])
	title = 'Roll distribution - %s pose' % label
	plt.title(title)
	plt.xlabel('angle (degrees)')
	plt.ylabel('number of occurrences')
	plt.savefig(os.path.join(PLOTS, title))

	plt.figure()
	pitchs_counts, pitchs_bins, _ = plt.hist(pitchs, bins=36, range = [-90, 90])
	title = 'Pitch distribution - %s pose' % label
	plt.title(title)
	plt.xlabel('angle (degrees)')
	plt.ylabel('number of occurrences')
	plt.savefig(os.path.join(PLOTS, title))

	y = get_y(yaws, yaw_bins, yaw_counts)
	y_all = np.concatenate((y_pain, np.asarray(y).reshape(-1,1)), axis = 1)
	output = Counter([tuple(i) for i in y_all])
	labels = np.vstack(output.keys())
	counts = np.vstack(output.values())
	y = np.vstack([np.where((labels == pain).all(axis = 1))for pain in y_all])
	max_labels = max(y)
	for i, index in enumerate(y):
		if counts[index] == 1:
			y[i] = max_labels[0] + 1

	train_X,test_X, train_y,  _ = train_test_split(x, y, test_size=0.30, random_state=42, stratify = y)

	train_angles = [angles[i] for i in range(len(angles)) if angles[i,0] in train_X]
	with open(os.path.join(ANGLES, '%s_pain_train_angles.pickle' % label), 'wb') as f:
		# Pickle the 'data' dictionary using the highest protocol available.
		pickle.dump(train_angles, f)

	test_angles = [angles[i] for i in range(len(angles)) if angles[i,0] in test_X]
	with open(os.path.join(ANGLES, '%s_pain_test_angles.pickle' % label), 'wb') as f:
		# Pickle the 'data' dictionary using the highest protocol available.
		pickle.dump(test_angles, f)

		skf = StratifiedKFold(n_splits = 3, random_state=42)
		i = 0
		for train_index, val_indexes in skf.split(train_X, train_y):
			fold = train_X[val_indexes]
			fold_angles = [angles[i] for i in range(len(angles)) if angles[i,0] in fold]
			with open(os.path.join(ANGLES, '%s_pain_val_fold_%d_angles.pickle' % (label, i)), 'wb') as f:
				# Pickle the 'data' dictionary using the highest protocol available.
				pickle.dump(fold_angles, f)
			i += 1
	return train_X, test_X

def make_copy(list_imgs, subfolder):
	for img in list_imgs:
		shutil.copy2(os.path.join(DATASET, 'images', img), os.path.join(CROSS_VAL, subfolder, img))

# %%============================================================================
#                                MAIN
# ==============================================================================

frontal_angles =  pickle.load(open(os.path.join(ANGLES, 'frontal_roll_pitch_yaw.pickle'), "rb"))
tilted_angles =  pickle.load(open(os.path.join(ANGLES, 'tilted_roll_pitch_yaw.pickle'), "rb"))
profile_angles =  pickle.load(open(os.path.join(ANGLES, 'profile_roll_pitch_yaw.pickle'), "rb"))

get_train_test(frontal_angles, 'frontal')
get_train_test(tilted_angles, 'tilted')
get_train_test(profile_angles, 'profile')


frontal_train_X,frontal_test_X = get_train_test(frontal_angles, 'frontal')
tilted_train_X,tilted_test_X = get_train_test(tilted_angles, 'tilted')
profile_train_X,profile_test_X = get_train_test(profile_angles, 'profile')

"""
make_copy(frontal_train_X, 'frontal/train')
make_copy(frontal_test_X, 'frontal/test')

make_copy(tilted_train_X, 'tilted/train')
make_copy(tilted_test_X, 'tilted/test')

make_copy(profile_train_X, 'profile/train')
make_copy(profile_test_X, 'profile/test')
"""