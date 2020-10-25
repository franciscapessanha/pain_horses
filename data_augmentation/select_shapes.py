#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 09:57:14 2020

@author: franciscapessanha
"""

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
#from pygem import RBFParameters, RBF, IDWParameters, IDW
import glob

DATASET = os.path.join(os.getcwd(), '..', 'dataset')

COLORS =  os.path.join(DATASET, '3D_annotations', 'colors')
SHAPES =  os.path.join(DATASET, '3D_annotations', 'shapes')

EX_FOLDER = os.path.join(os.getcwd(), '..', 'data_augmentation', 'examples')


all_colors = glob.glob(os.path.join(COLORS, '*.pickle'))
selected = glob.glob(os.path.join(EX_FOLDER, '*_base.png'))

selected = [i.split('/')[-1].split('_')[0] for i in selected]
all_colors = [i.split('/')[-1].split('.')[0].split('_')[-1] for i in all_colors]

for color in all_colors:
	if color not in selected:
		os.remove(os.path.join(COLORS, 'colors_' + color + '.pickle'))
		try:
			os.remove(os.path.join(SHAPES, color + '.obj'))
			os.remove(os.path.join(SHAPES, color + '.pickle'))
		except:
			print('done')