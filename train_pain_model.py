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
from pain_estimation.utils import *
from sklearn.metrics import f1_score, precision_recall_fscore_support
import random

PLOTS = os.path.join(os.getcwd(), 'plots')
if not os.path.exists(PLOTS):
    os.mkdir(PLOTS)


DATASET = os.path.join(os.getcwd(), 'dataset')
ANGLES =  os.path.join(DATASET, '3D_annotations', 'angles')

CROSS_VAL = os.path.join(os.getcwd(),'dataset','cross_val')
MODELS = os.path.join(os.getcwd(), 'pain_estimation','models')
N_FOLDS = 3

BIG_SIDE = 100

EX_FOLDER = os.path.join(os.getcwd(), 'pain_estimation', 'examples')

MODE = 'final_model' # final_model or cross_val
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

pain_scores = pd.read_excel(os.path.join(DATASET, 'pain_annotations.xlsx'), index_col=0, engine='openpyxl')

N_CLASSES = 2 #2 or 3
# %%============================================================================
#                            AUXILIAR FUNCTIONS
# ==============================================================================

def train_model(train_x, train_y, train_angles, train_rot, val_x, val_y, val_angles, val_rot, name_model,PPC, CPB, KERNEL,  O=9):

    train_x = [train_x[i] for i in range(len(train_y)) if train_y[i] != []]
    train_angles = [train_angles[i]for i in range(len(train_y)) if train_y[i] != []]
    train_angles = [[a[0][0], a[0][1], abs(a[0][2])] for a in train_angles]
    train_rot = [train_rot[i] for i in range(len(train_y)) if train_y[i] != []]
    train_y = np.hstack([train_y[i] for i in range(len(train_y)) if train_y[i] != []])

    val_x = [val_x[i] for i in range(len(val_y)) if val_y[i] != []]
    val_angles = [val_angles[i] for i in range(len(val_y)) if val_y[i] != []]
    val_angles = [[a[0][0], a[0][1], abs(a[0][2])] for a in val_angles]
    val_rot = [val_rot[i] for i in range(len(val_y)) if val_y[i] != []]
    val_y = np.hstack([val_y[i] for i in range(len(val_y)) if val_y[i] != []])



    HOGS_train, train_ratio = get_all_features(train_x, O, PPC, CPB)
    HOGS_train = np.asarray(HOGS_train)

    x_train = np.concatenate((np.vstack(HOGS_train), np.vstack(train_angles), np.vstack(train_rot)), axis = 1)
    print('train_y: ', len(train_y))
    train_y = np.hstack(train_y)
    train_y = [1 if y == 2 else y for y in train_y]

    modelSVM = SVC(kernel = KERNEL, gamma = 'auto', decision_function_shape= 'ovo',class_weight = 'balanced')
    modelSVM.fit(x_train, train_y)

    HOGS_val,  _= get_all_features(val_x, O, PPC, CPB, train_ratio)
    x_val = np.concatenate((np.vstack(HOGS_val), np.vstack(val_angles), np.vstack(val_rot)), axis = 1)
    y_pred = modelSVM.predict(x_val)

    if N_CLASSES == 2: val_y = [1 if y == 2 else y for y in val_y]

    elif N_CLASSES == 3:
        sample_weight = []
        for i in val_y:
            if i == 0:
                sample_weight.append(1)
            elif i ==1:
                sample_weight.append(2)
            elif i == 2:
                sample_weight.append(3)

    precision, recall, f1, _ = precision_recall_fscore_support(val_y, y_pred, average = 'weighted')

    target_names = ['0', '1']
    print(classification_report(val_y, y_pred, target_names=target_names))

    if list(val_y).count(1) > list(val_y).count(0):
        dummy = np.ones(np.shape(np.vstack(y_pred)))
    else:
        dummy = np.zeros(np.shape(np.vstack(y_pred)))
    #sample_weight = sample_weight
    print('Weighted MV F1_score: %0.2f' % f1_score(val_y, dummy, average = 'weighted'))

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
#                              MAIN
# ================================================================================
"""
comp_train_set = [] # complete training set (all validation sets)
for label in ['frontal', 'tilted', 'profile']:
    comp_train_set.append(np.load(open(os.path.join(ANGLES, '%s_pain_train_angles.pickle' % label), 'rb'), allow_pickle = True))

comp_train_set = np.vstack(comp_train_set)
ears, orbital, eyelid, sclera, nostrils, mouth = get_x_y(comp_train_set)

test_set = []
for label in ['frontal', 'tilted', 'profile']:
    test_set.append(np.load(open(os.path.join(ANGLES, '%s_pain_test_angles.pickle' % label), 'rb'), allow_pickle = True))

test_set = np.vstack(test_set)
ears_test, orbital_test, eyelid_test, sclera_test, nostrils_test, mouth_test = get_x_y(test_set)
"""
#%%

roi_labels = ['Ears', 'Orbital', 'Eyelid', 'Sclera', 'Nostrils', 'Mouth']
if MODE == 'cross_val':
    all_precision = []
    all_recall = []
    all_f1 = []

    for i,roi in enumerate([ears, orbital, eyelid, sclera, nostrils, mouth]):
        print(roi_labels[i])
        print('===============================')
        for KERNEL in ['linear', 'rbf']:
            for CPB in [1, 2, 3, 4]:
                for PPC in [8, 16]:
                    for k in range(N_FOLDS):
                        val_set = []
                        for label in ['frontal', 'tilted', 'profile']:
                            val_set.append(np.load(open(os.path.join(ANGLES, '%s_pain_val_fold_%d_angles.pickle' % (label, k)), 'rb'), allow_pickle = True))
                        val_set = np.vstack(val_set)

                        train_indexes = [i for i in range(len(comp_train_set)) if comp_train_set[i] not in val_set]
                        #:::::::::::::::::::::::::::::::::::::::::::
                        train, val = get_train_val(train_indexes, *roi)

                        precision, recall, f1 = train_model(*train, *val, '%s_bin_fold_%d_o_%d_ppc_%d_cpb_%d_kernel_%s' % (roi_labels[i], k, O, PPC, CPB, KERNEL), PPC, CPB, KERNEL)

                        all_precision.append(precision)
                        all_recall.append(recall)
                        all_f1.append(f1)

                    print('Kernel: ',  KERNEL, 'CPB: ', CPB, 'PPC: ', PPC)
                    print('Precision: %0.2f (+/-) %0.2f' % (np.mean(all_precision), np.std(all_precision)))
                    print('Recall: %0.2f (+/-) %0.2f' % (np.mean(all_recall), np.std(all_recall)))
                    print('F1-score: %0.2f (+/-) %0.2f' % (np.mean(all_f1), np.std(all_f1)))

elif MODE == 'final_model':
    # best option for each classifier [kernel, cpb, pcc]
    ears_best = ['linear', 3, 8]
    orbital_best = ['linear', 2, 8]
    eyelid_best = ['linear', 1, 8]
    sclera_best = ['linear', 2, 8]
    nostrils_best = ['linear', 4, 8]
    mouth_best = ['linear', 1, 8]
    all_parameters = [ears_best, orbital_best, eyelid_best, sclera_best, nostrils_best, mouth_best]
    for i,roi in enumerate([[ears, ears_test], [orbital, orbital_test], [eyelid, eyelid_test], [sclera, sclera_test], [nostrils, nostrils_test], [mouth, mouth_test]]):
        print(roi_labels[i])
        print('===============================')
        train = roi[0]
        test = roi[1]
        KERNEL, CPB, PPC = all_parameters[i]
        precision, recall, f1 = train_model(*train, *test, 'final_%s_bin_final_ppc_%d_cpb_%d_kernel_%s' % (roi_labels[i], PPC, CPB, KERNEL), PPC, CPB, KERNEL)


        print('Kernel: ',  KERNEL, 'CPB: ', CPB, 'PPC: ', PPC)
        print('Precision: %0.2f ' % precision)
        print('Recall: %0.2f ' % recall)
        print('F1-score: %0.2f ' % f1)
