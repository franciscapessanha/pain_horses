#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:54:21 2021

@author: Pessa001
"""

import os
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import pickle
import cv2 as cv
import trimesh
import glob
from data_augmentation.load_obj import *
from data_augmentation.transformations import *
import math
from utils import *
import random
import pandas as pd


columns = ['path', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'yaw', 'pitch', 'roll']


DATASET = os.path.join(os.getcwd(), 'dataset')
ANGLES =  os.path.join(DATASET, '3D_annotations', 'angles')

N_FOLDS = 3
alpha = 0.7
for  k in range(N_FOLDS):
    for AUG in [0.5, 0.7, 0.9]:
        path = os.path.join(os.getcwd(), 'dataset', 'pose', 'cropped_aug_data_alpha_%.1f_k_%d' % (AUG, k))
        if not os.path.exists(path):
            os.mkdir(path)


def save_dataframe(all_info, name):
    df_ = pd.DataFrame(columns=columns)
    for info in all_info:
        img_path = info[0]
        lms = np.vstack(info[1])
        try:
            img = cv.imread(img_path)
            lms_x = lms[:,0]
            lms_y = lms[:,1]

            img_h, img_w = img.shape[:2]
            x_min =  max(0,int(min(lms_x)))
            x_max = min(img_w, int(max(lms_x)))

            y_min = max(0, int(min(lms_y)))
            y_max = min(img_h, int(max(lms_y)))

            roll = float(info[2])
            pitch = float(info[3])
            yaw = float(info[4])

            dic = {'path': img_path,
                     'bbox_x_min': x_min,
                     'bbox_y_min': y_min,
                     'bbox_x_max': x_max,
                     'bbox_y_max': y_max,
                     'yaw': yaw,
                     'pitch': pitch,
                     'roll': roll}
            df_.loc[len(df_)] = dic
        except:
            pass
        df_.to_csv (name + '.csv', index = False, header=True)


for k in range(N_FOLDS):
    for alpha in [0.5, 0.7]:
        train = pd.read_csv('pose_aug/train_k_%d.csv' % k)

        list_folders = train['path'][0].split('/')[:8]

        list_ = glob.glob(os.path.join(DATASET, 'pose', 'aug_data_alpha_%.1f_final_new_formula_%d' % (alpha, k), '*.npy'))
        list_.sort()
        aug = []
        for info in list_:
            aug.append(np.load(info, allow_pickle=True))
        save_dataframe(aug, 'train_alpha_%.1f_final_k_%d' % (alpha, k))
        aug = pd.read_csv('train_alpha_%.1f_final_k_%d.csv' % (alpha, k))
        #%%
        for i, path in enumerate(aug['path']):
            """
            img = cv.imread(path)
            img_h, img_w, _ = np.shape(img)
            x_min = aug['bbox_x_min'][i]
            x_max = aug['bbox_x_max'][i]
            y_min = aug['bbox_y_min'][i]
            y_max = aug['bbox_y_max'][i]


            w = x_max - x_min
            h = y_max - y_min

            #aug['bbox_x_min'][i] = int(max(0, x_min - 0.05*w))
            #aug['bbox_x_max'][i] = int(min(x_max + 0.05*w, img_w))
            #aug['bbox_y_min'][i] = int(max(0, y_min - 0.05*h))
            #aug['bbox_y_max'][i] = int(min(y_max + 0.05*h, img_h))


            cv.circle(img, (int(x_min), int(y_min)), 5, (255,0,0), thickness=-1)
            cv.circle(img, (int(x_max), int(y_max)), 5, (255,0,0), thickness=-1)
            #img = img[aug['bbox_y_min'][i] : aug['bbox_y_max'][i], aug['bbox_x_min'][i]: aug['bbox_x_max'][i]]
            new_path = os.path.join('temp', path.split('/')[-1])

            cv.imwrite(new_path, img)
            """
            aug['path'][i] = '/' + os.path.join(*list_folders, *aug['path'][i].split('/')[7:])

        aug = pd.concat([aug, train])
        aug.to_csv ('pose_aug/corrected_train_alpha_%.1f_final_new_formula_k_%d.csv' % (alpha, k), index = False, header=True)
"""
train = pd.read_csv('pose_aug/train.csv')
aug = pd.read_csv('pose_aug/train_alpha_%.1f_final.csv' % (alpha))
list_folders = train['path'][0].split('/')[:8]
for i, path in enumerate(aug['path']):
    aug['path'][i] = '/' + os.path.join(*list_folders, *aug['path'][i].split('/')[7:])

aug = pd.concat([aug, train])
aug.to_csv ('pose_aug/corrected_train_alpha_%.1f_final.csv' % (alpha), index = False, header=True)
"""