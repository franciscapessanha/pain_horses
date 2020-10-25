#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 09:54:12 2020

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
from data_augmentation.utils import *
import math
import imutils
import numpy as np
import os
import glob
from skimage.feature import hog
import cv2 as cv
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pickle
from sklearn.metrics import confusion_matrix
from pain_estimation.get_features import *
from sklearn.metrics import f1_score, precision_recall_fscore_support
import random

PLOTS = os.path.join(os.getcwd(), 'plots')
if not os.path.exists(PLOTS):
	os.mkdir(PLOTS)


DATASET = os.path.join(os.getcwd(), 'dataset')
ANGLES =  os.path.join(DATASET, '3D_annotations', 'angles')

CROSS_VAL = os.path.join(os.getcwd(),'dataset','cross_val_pain')
MODELS = os.path.join(os.getcwd(), 'pain_estimation','models')
N_FOLDS = 3

BIG_SIDE = 100

EX_FOLDER = os.path.join(os.getcwd(), 'pain_estimation', 'examples')

if not os.path.exists(MODELS):
	os.mkdir(MODELS)

for folder in ['frontal', 'tilted','profile']:
	path = os.path.join(CROSS_VAL,folder)
	if not os.path.exists(path):
		os.mkdir(path)

	for sub_folder in ['train', 'test']:
		sub_path = os.path.join(path,sub_folder)
		if not os.path.exists(sub_path):
			os.mkdir(sub_path)

data = pickle.load(open(os.path.join(DATASET, 'lms_annotations.pkl'), "rb"))

pain_scores = pd.read_excel(os.path.join(DATASET, 'pain_annotations.xlsx'), index_col=0)


kernel = 'linear'
O = 9
PPC = 8
CPB = 4

def flip_image(img, lms, pose):
	if pose < 0:
		img = cv.flip(img, 1)
		h, w = np.shape(img)[:2]
		mirror_lms = []
		for pt in lms:
			new_pt = [w - pt[0], pt[1]]
			mirror_lms.append(new_pt)

		lms = np.vstack(mirror_lms)

	return img,lms

# ==============================================================================
def flat_list(list_of_lists):
	flat_list = [item for sublist in list_of_lists for item in sublist if sublist != []]
	return flat_list
def get_x_y(set_):
	ears_x = []
	ears_rot = []
	ears_angles = []
	ears_y = []

	orbital_x = []
	orbital_rot = []
	orbital_angles = []
	orbital_y = []

	eyelid_x = []
	eyelid_rot = []
	eyelid_angles = []
	eyelid_y = []

	sclera_x = []
	sclera_rot = []
	sclera_angles = []
	sclera_y = []

	nostrils_x = []
	nostrils_rot = []
	nostrils_angles = []
	nostrils_y = []

	mouth_x = []
	mouth_rot = []
	mouth_angles = []
	mouth_y = []

	for j, angles in enumerate(set_):
		#print(angles)
		#print(j + 1, '/', len(set_))
		index = int(angles[0].split('.')[0])
		info = data.values[index - 1]
		#print(os.path.join(os.getcwd(), info[0]))
		img = cv.imread(os.path.join(os.getcwd(), info[0]))
		lms = info[-1]
		pose = info[2]

		pain_info = pain_scores.values[index][1:7]

		img,lms =  flip_image(img, lms, pose)
		lms = update_landmarks(lms, pose)

		#cv.imwrite(os.path.join(EX_FOLDER, str(random.randint(0,10000)) +  '.jpg'), img)
		#ears, eyes, eyes, eyes, mouth, nostrils
		eye, eye_rot = get_eyes(img, lms, pose)
		for i,pain in enumerate(pain_info):
				if i == 0:
					if pain == -1:
						ears_x.append([])
						ears_rot.append([])
						ears_y.append([])
						ears_angles.append([])
					else:
						ears, rot = get_ears(img, lms, pose)
						ears_x.append(ears)
						ears_rot.append(rot)
						y = []
						a = []
						#for j in range(len(ears_x[-1])):
						for j in range(1):
							y.append(pain)
							a.append(angles[1:].astype(np.float))
						ears_y.append(y)
						ears_angles.append(a)

				elif i == 1:
					if pain == -1:
						orbital_x.append([])
						orbital_rot.append([])
						orbital_y.append([])
						orbital_angles.append([])
					else:
						orbital_x.append(eye)
						orbital_rot.append(eye_rot)
						y = []
						a = []
						for j in range(len(orbital_x[-1])):
							y.append(pain)
							a.append(angles[1:].astype(np.float))
						orbital_y.append(y)
						orbital_angles.append(a)

				elif i == 2:
					if pain == -1:
						eyelid_x.append([])
						eyelid_rot.append([])
						eyelid_y.append([])
						eyelid_angles.append([])
					else:
						eyelid_x.append(eye)
						eyelid_rot.append(eye_rot)
						y = []
						a = []
						for j in range(len(eyelid_x[-1])):
							y.append(pain)
							a.append(angles[1:].astype(np.float))
						eyelid_y.append(y)
						eyelid_angles.append(a)

				elif i == 3:
						if pain == -1:
							sclera_x.append([])
							sclera_rot.append([])
							sclera_y.append([])
							sclera_angles.append([])
						else:
							sclera_x.append(eye)
							sclera_rot.append(eye_rot)
							y = []
							a = []
							#for j in range(len(sclera_x[-1])):
							for j in range(1):
								y.append(pain)
								a.append(angles[1:].astype(np.float))
							sclera_y.append(y)
							sclera_angles.append(a)

				elif i == 4:
					if pain == -1 or pose == 0:
						 mouth_x.append([])
						 mouth_rot.append([])
						 mouth_y.append([])
						 mouth_angles.append([])

					else:
						mouth, rot = get_mouth(img, lms, pose)
						mouth_x.append([mouth])
						mouth_rot.append([rot])
						mouth_angles.append([angles[1:].astype(np.float)])
						mouth_y.append(pain)

				elif i == 5:
					if pain == -1:
						nostrils_x.append([])
						nostrils_rot.append([])
						nostrils_y.append([])
						nostrils_angles.append([])

					else:
						nostrils, rot = get_nostrils(img, lms, pose)
						nostrils_x.append(nostrils)
						nostrils_rot.append(rot)
						y = []
						a = []
						#for j in range(len(nostrils_x[-1])):
						for j in range(1):
							y.append(pain)
							a.append(angles[1:].astype(np.float))
							#print(angles[1:])
						nostrils_y.append(y)
						nostrils_angles.append(a)


	return [ears_x, ears_y, ears_angles, ears_rot],[orbital_x, orbital_y, orbital_angles, orbital_rot],[eyelid_x, eyelid_y, eyelid_angles, eyelid_rot], [sclera_x, sclera_y, sclera_angles, sclera_rot], [nostrils_x, nostrils_y, nostrils_angles, nostrils_rot], [mouth_x, mouth_y, mouth_angles, mouth_rot]

def train_model(train_x, train_y, train_angles, train_rot, val_x, val_y, val_angles, val_rot, name_model):

	train_x = [train_x[i] for i in range(len(train_y)) if train_y[i] != []]
	train_angles = [train_angles[i]for i in range(len(train_y)) if train_y[i] != []]
	train_angles = [[a[0][0], a[0][1], abs(a[0][2])] for a in train_angles]
	train_rot = [train_rot[i] for i in range(len(train_y)) if train_y[i] != []]
	train_y = [train_y[i] for i in range(len(train_y)) if train_y[i] != []]

	val_x = [val_x[i] for i in range(len(val_y)) if val_y[i] != []]
	val_angles = [val_angles[i] for i in range(len(val_y)) if val_y[i] != []]
	val_angles = [[a[0][0], a[0][1], abs(a[0][2])] for a in val_angles]
	val_rot = [val_rot[i] for i in range(len(val_y)) if val_y[i] != []]
	val_y = [val_y[i] for i in range(len(val_y)) if val_y[i] != []]


	HOGS_train, train_ratio = get_all_features(train_x, O, PPC, CPB)

	#print('HOGS: ', len(HOGS_train))
	#print('y : ', len(train_y))
	#print('angles : ', len(train_angles))
	HOGS_train = np.asarray(HOGS_train)

	#print(HOGS_train[:, 1].reshape(-1,1))
	x_train = np.concatenate((np.vstack(HOGS_train), np.vstack(train_angles), np.vstack(train_rot)), axis = 1)
	print(np.shape(np.vstack(x_train)))
	train_y = [1 if y[0] == 2 else y[0] for y in train_y]
	#train_y = [y for y in train_y]

	modelSVM = SVC(kernel = kernel, gamma = 'auto', decision_function_shape= 'ovo',class_weight = 'balanced')
	modelSVM.fit(x_train, train_y)

	HOGS_val,  _= get_all_features(val_x, O, PPC, CPB, train_ratio)
	x_val = np.concatenate((np.vstack(HOGS_val), np.vstack(val_angles), np.vstack(val_rot)), axis = 1)
	y_pred = modelSVM.predict(x_val)
	val_y = [1 if y[0] == 2 else y[0] for y in val_y]
	#val_y = [y[0] for y in val_y]
	sample_weight = []
	#weight_0 = 1 / (list(train_y).count(0)/len(train_y))
	#weight_1 = 1 / (list(train_y).count(1)/len(train_y))
	#weight_2 = 1 / (list(train_y).count(2)/len(train_y))

	for i in val_y:
		if i == 0:
			sample_weight.append(1)
		elif i ==1:
			sample_weight.append(2)
		elif i == 2:
			sample_weight.append(3)

	precision, recall, f1, _ = precision_recall_fscore_support(val_y, y_pred, average = 'micro')

	#print('Pred: ', y_pred)
	#print('Real: ', val_y)

	#print('precision: ', precision)
	#print('recall: ', recall)
	#print('f1_score: ', f1)
	target_names = ['0', '1']
	print(classification_report(val_y, y_pred, target_names=target_names))

	if list(val_y).count(1) > list(val_y).count(0):
		dummy = np.ones(np.shape(np.vstack(y_pred)))
	else:
		dummy = np.zeros(np.shape(np.vstack(y_pred)))
	#sample_weight = sample_weight
	print('MV F1_score: %0.2f' % f1_score(val_y, dummy, average = 'micro'))

	with open(os.path.join(MODELS, name_model + '.pickle'), 'wb') as f:
		# Pickle the 'data' dictionary using the highest protocol available.
		pickle.dump(modelSVM, f)

	return precision, recall, f1

def get_train_val(train_indexes, all_x, all_y, all_angles, all_rot):
	train_x = [all_x[i] for i in train_indexes]
	val_x = [all_x[i] for i in range(len(all_x)) if i not in train_indexes]

	train_y = [all_y[i] for i in train_indexes]
	val_y = [all_y[i] for i in range(len(all_y)) if i not in train_indexes]

	train_pose = [all_angles[i] for i in train_indexes]
	val_pose = [all_angles[i] for i in range(len(all_y)) if i not in train_indexes]

	train_rot = [all_rot[i] for i in train_indexes]
	val_rot = [all_rot[i] for i in range(len(all_y)) if i not in train_indexes]
	return [train_x, train_y, train_pose, train_rot], [val_x, val_y, val_pose, val_rot]

#%% ==============================================================================

comp_train_set = []
for label in ['frontal', 'tilted', 'profile']:
	comp_train_set.append(np.load(open(os.path.join(ANGLES, '%s_pain_train_angles.pickle' % label), 'rb'), allow_pickle = True))

comp_train_set = np.vstack(comp_train_set)
ears, orbital, eyelid, sclera, nostrils, mouth = get_x_y(comp_train_set)


test_set = []
for label in ['frontal', 'tilted', 'profile']:
	test_set.append(np.load(open(os.path.join(ANGLES, '%s_pain_test_angles.pickle' % label), 'rb'), allow_pickle = True))

test_set = np.vstack(test_set)
ears_test, orbital_test, eyelid_test, sclera_test, nostrils_test, mouth_test = get_x_y(test_set)
#%%
"""
all_precision = []
all_recall = []
all_f1 = []
print('Ears')
for KERNEL in ['linear']:
	for CPB in [3]:
		for PPC in [8]:
			for k in range(N_FOLDS):
				val_set = []
				for label in ['frontal', 'tilted', 'profile']:
					val_set.append(np.load(open(os.path.join(ANGLES, '%s_pain_val_fold_%d_angles.pickle' % (label, k)), 'rb'), allow_pickle = True))
				val_set = np.vstack(val_set)

				train_indexes = [i for i in range(len(comp_train_set)) if comp_train_set[i] not in val_set]
				#:::::::::::::::::::::::::::::::::::::::::::
				ears_train, ears_val = get_train_val(train_indexes, *ears)

				precision, recall, f1 = train_model(*ears_train, *ears_val, 'ear_bin_fold_%d_o_%d_ppc_%d_cpb_%d_kernel_%s' % (k, O, PPC, CPB, KERNEL))

				all_precision.append(precision)
				all_recall.append(recall)
				all_f1.append(f1)

			print('Kernel: ',  KERNEL, 'CPB: ', CPB, 'PPC: ', PPC)
			print('Precision: %0.2f (+/-) %0.2f' % (np.mean(all_precision), np.std(all_precision)))
			print('Recall: %0.2f (+/-) %0.2f' % (np.mean(all_recall), np.std(all_recall)))
			print('F1-score: %0.2f (+/-) %0.2f' % (np.mean(all_f1), np.std(all_f1)))
"""
#%%
print('Ears')
all_precision = []
all_recall = []
all_f1 = []
all_acc = []
for KERNEL in ['linear']:
	for CPB in [3]:
		for PPC in [8]:
			precision, recall, f1 = train_model(*ears, *ears_test, 'final_ear_bin_fold_%d_o_%d_ppc_%d_cpb_%d_kernel_%s' % (k, O, PPC, CPB, KERNEL))

			all_precision.append(precision)

			all_recall.append(recall)
			all_f1.append(f1)

			print('Kernel: ',  KERNEL, 'CPB: ', CPB, 'PPC: ', PPC)
			print('Precision: %0.2f (+/-) %0.2f' % (np.mean(all_precision), np.std(all_precision)))
			print('Recall: %0.2f (+/-) %0.2f' % (np.mean(all_recall), np.std(all_recall)))
			print('F1-score: %0.2f (+/-) %0.2f' % (np.mean(all_f1), np.std(all_f1)))



"""
all_precision = []
all_recall = []
all_f1 = []
all_acc = []
print('Orbital')
for KERNEL in ['linear']:
	for CPB in [1, 2, 3, 4]:
		for PPC in [8]:
			for k in range(N_FOLDS):
				val_set = []
				for label in ['frontal', 'tilted', 'profile']:
					val_set.append(np.load(open(os.path.join(ANGLES, '%s_pain_val_fold_%d_angles.pickle' % (label, k)), 'rb'), allow_pickle = True))
				val_set = np.vstack(val_set)

				train_indexes = [i for i in range(len(comp_train_set)) if comp_train_set[i] not in val_set]
				#:::::::::::::::::::::::::::::::::::::::::::

				orbital_train, orbital_val = get_train_val(train_indexes, *orbital)
				precision, recall, f1 = train_model(*orbital_train, *orbital_val, 'orbital_bin_fold_%d_o_%d_ppc_%d_cpb_%d_kernel_%s' % (k, O, PPC, CPB, KERNEL))

				all_precision.append(precision)
				all_recall.append(recall)
				all_f1.append(f1)


			print('Kernel: ',  KERNEL, 'CPB: ', CPB, 'PPC: ', PPC)
			print('Precision: %0.2f (+/-) %0.2f' % (np.mean(all_precision), np.std(all_precision)))
			print('Recall: %0.2f (+/-) %0.2f' % (np.mean(all_recall), np.std(all_recall)))
			print('F1-score: %0.2f (+/-) %0.2f' % (np.mean(all_f1), np.std(all_f1)))
"""

all_precision = []
all_recall = []
all_f1 = []
all_acc = []
print('Orbital')
for KERNEL in ['linear']:
	for CPB in [2]:
		for PPC in [8]:
			precision, recall, f1 = train_model(*orbital, *orbital_test, 'final_orbital_bin_fold_%d_o_%d_ppc_%d_cpb_%d_kernel_%s' % (k, O, PPC, CPB, KERNEL))

			all_precision.append(precision)
			all_recall.append(recall)
			all_f1.append(f1)

			print('Kernel: ',  KERNEL, 'CPB: ', CPB, 'PPC: ', PPC)
			print('Precision: %0.2f (+/-) %0.2f' % (np.mean(all_precision), np.std(all_precision)))
			print('Recall: %0.2f (+/-) %0.2f' % (np.mean(all_recall), np.std(all_recall)))
			print('F1-score: %0.2f (+/-) %0.2f' % (np.mean(all_f1), np.std(all_f1)))


"""
all_precision = []
all_recall = []
all_f1 = []
all_acc = []
print('Eyelid')
for KERNEL in ['linear']:
	for CPB in [1, 2, 3, 4]:
		for PPC in [8]:
			for k in range(N_FOLDS):
				val_set = []
				for label in ['profile']:
					val_set.append(np.load(open(os.path.join(ANGLES, '%s_pain_val_fold_%d_angles.pickle' % (label, k)), 'rb'), allow_pickle = True))
				val_set = np.vstack(val_set)

				train_indexes = [i for i in range(len(comp_train_set)) if comp_train_set[i] not in val_set]
				#:::::::::::::::::::::::::::::::::::::::::::

				eyelid_train, eyelid_val = get_train_val(train_indexes, *eyelid)
				precision, recall, f1 = train_model(*eyelid_train, *eyelid_val, 'eyelid_bin_fold_%d_o_%d_ppc_%d_cpb_%d_kernel_%s' % (k, O, PPC, CPB, KERNEL))

				all_precision.append(precision)
				all_recall.append(recall)
				all_f1.append(f1)


			print('Kernel: ',  KERNEL, 'CPB: ', CPB, 'PPC: ', PPC)
			print('Precision: %0.2f (+/-) %0.2f' % (np.mean(all_precision), np.std(all_precision)))
			print('Recall: %0.2f (+/-) %0.2f' % (np.mean(all_recall), np.std(all_recall)))
			print('F1-score: %0.2f (+/-) %0.2f' % (np.mean(all_f1), np.std(all_f1)))
"""

all_precision = []
all_recall = []
all_f1 = []
all_acc = []
print('Eyelid')
for KERNEL in ['linear']:
	for CPB in [1]:
		for PPC in [8]:
			precision, recall, f1 = train_model(*eyelid, *eyelid_test, 'final_eyelid_bin_fold_%d_o_%d_ppc_%d_cpb_%d_kernel_%s' % (k, O, PPC, CPB, KERNEL))

			all_precision.append(precision)
			all_recall.append(recall)
			all_f1.append(f1)

			print('Kernel: ',  KERNEL, 'CPB: ', CPB, 'PPC: ', PPC)
			print('Precision: %0.2f (+/-) %0.2f' % (np.mean(all_precision), np.std(all_precision)))
			print('Recall: %0.2f (+/-) %0.2f' % (np.mean(all_recall), np.std(all_recall)))
			print('F1-score: %0.2f (+/-) %0.2f' % (np.mean(all_f1), np.std(all_f1)))



"""
all_precision = []
all_recall = []
all_f1 = []
all_acc = []
print('Sclera')

for KERNEL in ['linear']:
	for CPB in [1, 2, 3, 4]:
		for PPC in [8]:
			for k in range(N_FOLDS):
				val_set = []
				for label in ['frontal', 'tilted', 'profile']:
					val_set.append(np.load(open(os.path.join(ANGLES, '%s_pain_val_fold_%d_angles.pickle' % (label, k)), 'rb'), allow_pickle = True))
				val_set = np.vstack(val_set)

				train_indexes = [i for i in range(len(comp_train_set)) if comp_train_set[i] not in val_set]
				sclera_train, sclera_val = get_train_val(train_indexes, *sclera)
				precision, recall, f1 = train_model(*sclera_train, *sclera_val, 'sclera_bin_fold_%d_o_%d_ppc_%d_cpb_%d_kernel_%s' % (k, O, PPC, CPB, KERNEL))

				all_precision.append(precision)
				all_recall.append(recall)
				all_f1.append(f1)

			print('Kernel: ',  KERNEL, 'CPB: ', CPB, 'PPC: ', PPC)
			print('Precision: %0.2f (+/-) %0.2f' % (np.mean(all_precision), np.std(all_precision)))
			print('Recall: %0.2f (+/-) %0.2f' % (np.mean(all_recall), np.std(all_recall)))
			print('F1-score: %0.2f (+/-) %0.2f' % (np.mean(all_f1), np.std(all_f1)))
"""

all_precision = []
all_recall = []
all_f1 = []
all_acc = []
print('Sclera')
for KERNEL in ['linear']:
	for CPB in [2]:
		for PPC in [8]:
			precision, recall, f1 = train_model(*sclera, *sclera_test, 'final_sclera_bin_fold_%d_o_%d_ppc_%d_cpb_%d_kernel_%s' % (k, O, PPC, CPB, KERNEL))

			all_precision.append(precision)
			all_recall.append(recall)
			all_f1.append(f1)

			print('Kernel: ',  KERNEL, 'CPB: ', CPB, 'PPC: ', PPC)
			print('Precision: %0.2f (+/-) %0.2f' % (np.mean(all_precision), np.std(all_precision)))
			print('Recall: %0.2f (+/-) %0.2f' % (np.mean(all_recall), np.std(all_recall)))
			print('F1-score: %0.2f (+/-) %0.2f' % (np.mean(all_f1), np.std(all_f1)))




"""
all_precision = []
all_recall = []
all_f1 = []
all_acc = []
print('Mouth')

for KERNEL in ['linear']:
	for CPB in [1]:
		for PPC in [8]:
			for k in range(N_FOLDS):
				val_set = []
				for label in ['frontal', 'tilted', 'profile']:
					val_set.append(np.load(open(os.path.join(ANGLES, '%s_pain_val_fold_%d_angles.pickle' % (label, k)), 'rb'), allow_pickle = True))
				val_set = np.vstack(val_set)

				train_indexes = [i for i in range(len(comp_train_set)) if comp_train_set[i] not in val_set]
				mouth_train, mouth_val = get_train_val(train_indexes, *mouth)
				precision, recall, f1 = train_model(*mouth_train, *mouth_val, 'mouth_bin_fold_%d_o_%d_ppc_%d_cpb_%d_kernel_%s' % (k, O, PPC, CPB, KERNEL))

				all_precision.append(precision)
				all_recall.append(recall)
				all_f1.append(f1)

			print('Kernel: ',  KERNEL, 'CPB: ', CPB, 'PPC: ', PPC)
			print('Precision: %0.2f (+/-) %0.2f' % (np.mean(all_precision), np.std(all_precision)))
			print('Recall: %0.2f (+/-) %0.2f' % (np.mean(all_recall), np.std(all_recall)))
			print('F1-score: %0.2f (+/-) %0.2f' % (np.mean(all_f1), np.std(all_f1)))

#%%
all_precision = []
all_recall = []
all_f1 = []
all_acc = []
print('Mouth')
for KERNEL in ['linear']:
	for CPB in [1]:
		for PPC in [8]:
			precision, recall, f1 = train_model(*mouth, *mouth_test, 'final_mouth_bin_fold_%d_o_%d_ppc_%d_cpb_%d_kernel_%s' % (k, O, PPC, CPB, KERNEL))

			all_precision.append(precision)
			all_recall.append(recall)
			all_f1.append(f1)

			print('Kernel: ',  KERNEL, 'CPB: ', CPB, 'PPC: ', PPC)
			print('Precision: %0.2f (+/-) %0.2f' % (np.mean(all_precision), np.std(all_precision)))
			print('Recall: %0.2f (+/-) %0.2f' % (np.mean(all_recall), np.std(all_recall)))
			print('F1-score: %0.2f (+/-) %0.2f' % (np.mean(all_f1), np.std(all_f1)))



#%%
all_precision = []
all_recall = []
all_f1 = []
all_acc = []
print('Nostrils')

for KERNEL in ['linear']:
	for CPB in [1, 2, 3, 4]:
		for PPC in [8]:
			for k in range(N_FOLDS):
				val_set = []
				for label in ['frontal', 'tilted', 'profile']:
					val_set.append(np.load(open(os.path.join(ANGLES, '%s_pain_val_fold_%d_angles.pickle' % (label, k)), 'rb'), allow_pickle = True))
				val_set = np.vstack(val_set)

				train_indexes = [i for i in range(len(comp_train_set)) if comp_train_set[i] not in val_set]
				nostrils_train, nostrils_val = get_train_val(train_indexes, *nostrils)
				precision, recall, f1 = train_model(*nostrils_train, *nostrils_val, 'nostrils_bin_fold_%d_o_%d_ppc_%d_cpb_%d_kernel_%s' % (k, O, PPC, CPB, KERNEL))

				all_precision.append(precision)
				all_recall.append(recall)
				all_f1.append(f1)

			print('Kernel: ',  KERNEL, 'CPB: ', CPB, 'PPC: ', PPC)
			print('Precision: %0.2f (+/-) %0.2f' % (np.mean(all_precision), np.std(all_precision)))
			print('Recall: %0.2f (+/-) %0.2f' % (np.mean(all_recall), np.std(all_recall)))
			print('F1-score: %0.2f (+/-) %0.2f' % (np.mean(all_f1), np.std(all_f1)))
"""

all_precision = []
all_recall = []
all_f1 = []
all_acc = []
print('Nostrils')
for KERNEL in ['linear']:
	for CPB in [4]:
		for PPC in [8]:
			precision, recall, f1 = train_model(*sclera, *sclera_test, 'final_nostrils_bin_fold_%d_o_%d_ppc_%d_cpb_%d_kernel_%s' % (k, O, PPC, CPB, KERNEL))

			all_precision.append(precision)
			all_recall.append(recall)
			all_f1.append(f1)

			print('Kernel: ',  KERNEL, 'CPB: ', CPB, 'PPC: ', PPC)
			print('Precision: %0.2f (+/-) %0.2f' % (np.mean(all_precision), np.std(all_precision)))
			print('Recall: %0.2f (+/-) %0.2f' % (np.mean(all_recall), np.std(all_recall)))
			print('F1-score: %0.2f (+/-) %0.2f' % (np.mean(all_f1), np.std(all_f1)))



