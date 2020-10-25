#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 13:36:17 2020
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
#from pygem import RBFParameters, RBF, IDWParameters, IDW
import glob
import random
from scipy.spatial.transform import Rotation as Rot


DATASET = os.path.join(os.getcwd(), 'dataset')
EX_FOLDER = os.path.join(os.getcwd(), 'examples')
if not os.path.exists(EX_FOLDER):
	os.mkdir(EX_FOLDER)

data = pickle.load(open(os.path.join(DATASET, 'lms_annotations.pkl'), "rb"))

def find_closest_edge(edges, pt):
	all_dist = []
	for y in range(np.shape(edges)[0]):
		for x in range(np.shape(edges)[1]):
			if edges[y,x] ==  255:
				dist = np.sqrt((pt[0] - x)**2 + (pt[1] - y)**2)
				all_dist.append([x, y, dist])

	all_dist = np.vstack(all_dist)
	index = np.argmin(all_dist[-1,:], axis=0)

	return [all_dist[index][0], all_dist[index][1]]


#%%
def update_landmarks(info):
	if abs(info[2]) == 60:
		outline_lms_ind = [1, 2, 3, 4, 5, 40, 41, 42, 43, 44, 45]
		outline_lms_ind = [lms - 1 for lms in outline_lms_ind]
		outline_lms = [info[-1][i] for i in outline_lms_ind]

		img = cv.imread(os.path.join(os.getcwd(),info[0]))
		h, w = np.shape(img)[:2]

		new_pts = []
		img = cv.GaussianBlur(img,(9,9),0)


		#---- Apply automatic Canny edge detection using the computed median----

		#cv.imwrite(os.path.join(EX_FOLDER, 'edges_' + info[0].split('/')[-1]), edges)
		for j, pt in enumerate(outline_lms):
			x_min = int(max(0, pt[0] - 0.005*w))
			x_max = int(min(w, pt[0] + 0.005*w))
			y_min = int(max(0, pt[1] - 0.005*h))
			y_max = int(min(h, pt[1] + 0.05*h)) # this is the direction we want to move into
			crop_img = img[y_min:y_max,x_min:x_max,:]
			gray_image = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)
			sigma = 0.33
			v = np.median(gray_image)
			lower = int(max(0, (1.0 - sigma) * v))
			upper = int(min(255, (1.0 + sigma) * v))
			edges = cv.Canny(gray_image, lower, upper)

			new_pt = ((pt[0] - x_min), (pt[1] - y_min))
			#cv.imwrite(os.path.join(EX_FOLDER, 'edges_%d' % j + info[0].split('/')[-1]), edges)
			#cv.imwrite(os.path.join(EX_FOLDER, 'crop_%d' % j + info[0].split('/')[-1]), crop_img)
			#PROFILE_TRAIN = glob.glob(os.path.join(DATASET, 'cross_val', 'profile', 'train', '*.jpg'))
			if edges[int(new_pt[1]),int(new_pt[0])] == 0:
				if max(np.hstack(edges)) != 0:
					update_point = find_closest_edge(edges, new_pt)
					new_pts.append(tuple([update_point[0] + x_min, update_point[1] + y_min]))
				else:
					new_pts.append(tuple([pt[0], pt[1]]))
			else:
				new_pts.append(tuple([pt[0], pt[1]]))

		"""
		k = 0
		for i, pt in enumerate(info[-1]):
			if i in outline_lms_ind:
				cv.circle(img,tuple((int(pt[0]), int(pt[1]))), 3, (255,0,0), -1)
				new_pt = new_pts[k]
				k += 1
				cv.circle(img,tuple((int(new_pt[0]), int(new_pt[1]))), 3, (0,0,255), -1)
				cv.imwrite(os.path.join(EX_FOLDER, 'update_' + info[0].split('/')[-1]), img)
		"""
		lms = info[-1].copy()
		for i, index in enumerate(outline_lms_ind):
			lms[index] = new_pts[i]

		return lms
