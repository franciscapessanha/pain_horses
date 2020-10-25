#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:25:31 2020

@author: franciscapessanha
"""

#%%============================================================================
#                       IMPORTS AND INITIALIZATIONS
#%============================================================================
import numpy as np
import os
import pickle
import cv2 as cv
import shutil
import random

random.seed(10)

DATASET = os.path.join(os.getcwd(), '../dataset')

        
data = pickle.load(open( os.path.join(DATASET, 'lms_annotations.pkl'), "rb" ))


lms_30 = []
indexes_30 = []
for i, info in enumerate(data.values):
    if abs(info[2]) == 30 and info[1] == 'horse':
        lms_30.append(len(info[-1]))
        if len(info[-1]) != 44: 
            indexes_30.append(i) 
            img = cv.imread(info[0])
            for k,pt in enumerate(info[-2][30:]):
                r = 5
                cv.circle(img, (int(pt[0]), int(pt[1])), k+6, (5,0,0), -1)
                    
            #cv.imwrite(os.path.join(DATASET,'30_test_norm.jpg'), img)

lms_60 = []
indexes_60 = []
for i, info in enumerate(data.values):
    if abs(info[2]) == 60 and info[1] == 'horse':
        lms_60.append(len(info[-1]))
        if len(info[-1]) != 45: 
            indexes_60.append(i) 
            img = cv.imread(info[0])
            for k,pt in enumerate(info[-2][30:]):
                r = 5
                cv.circle(img, (int(pt[0]), int(pt[1])), k+6, (k*5,0,0), -1)
                
                #cv.imwrite(os.path.join(DATASET,'60_test.jpg'), img)
                #cv.imshow('original', img)
                #cv.waitKey(0)
                #cv.destroyAllWindows()
            
lms_0 = []
indexes_0 = []
for i, info in enumerate(data.values):
    if abs(info[2]) == 0 and info[1] == 'horse':
        lms_0.append(len(info[-1]))
        if len(info[-1]) != 54: 
            indexes_0.append(i) 
            