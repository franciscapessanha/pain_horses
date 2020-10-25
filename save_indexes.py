#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 12:33:53 2020

@author: franciscapessanha
"""

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from data_augmentation.ObjLoader import *
import numpy as np
import os
import pickle
import cv2 as cv
from mpl_toolkits.mplot3d import Axes3D
import glob
from data_augmentation.load_obj import *
from data_augmentation.transformations import *
from data_augmentation.utils import *

DATASET = os.path.join(os.getcwd(), 'dataset')
MODELS =  os.path.join(DATASET, '3D_annotations', 'models')
LANDMARKS =  os.path.join(DATASET, '3D_annotations', 'landmarks')
ANGLES =  os.path.join(DATASET, '3D_annotations', 'angles')

HEAD_OBJ = os.path.join(MODELS, 'head_TOSCA.obj')

TILTED_OUTLINE = os.path.join(LANDMARKS, 'tilted_outline.txt')
TILTED_EARS = os.path.join(LANDMARKS, 'tilted_ears.txt')
TILTED_LEFT_NOSTRIL = os.path.join(LANDMARKS, 'tilted_left_nostril.txt')

FRONTAL_LEFT_EAR = os.path.join(LANDMARKS, 'frontal_left_ear.txt')
FRONTAL_OUTLINE = os.path.join(LANDMARKS, 'frontal_outline.txt')
FRONTAL_LEFT_NOSTRIL = os.path.join(LANDMARKS, 'frontal_left_nostril.txt')

PROFILE_LEFT_EAR = os.path.join(LANDMARKS, 'profile_ear.txt')
PROFILE_OUTLINE = os.path.join(LANDMARKS, 'profile_all_outline.txt')
PROFILE_LEFT_NOSTRIL = os.path.join(LANDMARKS, 'profile_left_nostril.txt')

CHEEK = os.path.join(LANDMARKS, 'left_cheek.txt')
MOUTH = os.path.join(LANDMARKS, 'mouth.txt')

LEFT_EYE = os.path.join(LANDMARKS, 'left_eye.txt')


# %%============================================================================
#                                MAIN
# ==============================================================================

vertices, triangles = load_obj(HEAD_OBJ)
correspondence, zeros =  symetric(vertices)

tilted_outline = load_lms_model(TILTED_OUTLINE)
tilted_ears = load_lms_model(TILTED_EARS)
tilted_left_nostril = load_lms_model(TILTED_LEFT_NOSTRIL)

left_ear = load_lms_model(FRONTAL_LEFT_EAR)
frontal_outline = load_lms_model(FRONTAL_OUTLINE)
frontal_left_nostril = load_lms_model(FRONTAL_LEFT_NOSTRIL)

profile_ear = load_lms_model(PROFILE_LEFT_EAR)
profile_outline = load_lms_model(PROFILE_OUTLINE)
profile_left_nostril = load_lms_model(PROFILE_LEFT_NOSTRIL)

cheek = load_lms_model(CHEEK)
mouth = load_lms_model(MOUTH)

left_eye = load_lms_model(LEFT_EYE)

#for frontal
right_ear = [[pt[0], -pt[1], pt[2], pt[3]] for pt in left_ear]
right_ear = [right_ear[i] for i in range(len(right_ear)-1, -1, -1)]
right_eye = [[pt[0], -pt[1], pt[2], pt[3]] for pt in left_eye]
right_eye = [right_eye[i] for i in [2, 1, 0, 5, 4, 3]]
right_nostril = [[pt[0], -pt[1], pt[2], pt[3]] for pt in frontal_left_nostril]
right_nostril = [right_nostril[i] for i in [2, 1, 0, 5, 4, 3]]

model_shape = np.concatenate((right_ear, left_ear, left_eye, frontal_left_nostril, right_nostril, right_eye, frontal_outline), axis = 0)[:,1:] #ears are not included because they are too variable
indexes = np.vstack(get_indexes(vertices, model_shape))
sim_indexes = indexes.copy()

for j, i in enumerate(indexes):
	if i in zeros:
		sim_indexes[j] = zeros[np.where(zeros == i)[0][0]]

	elif i in correspondence[:,0]:
		sim_indexes[j] = correspondence[np.where(correspondence[:,0] == i)[0][0],1]

	elif i in correspondence[:,1]:
		sim_indexes[j] = correspondence[np.where(correspondence[:,1] == i)[0][0],0]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
vert = np.vstack([vertices[i] for i in indexes])
ax.scatter(vert[:, 0], vert[:, 1], vert[:, 2], c = 'r', alpha = 0.1)
ax.scatter(model_shape[:, 0], model_shape[:, 1], model_shape[:, 2], c = 'b')
plt.show()

with open(os.path.join(MODELS, 'frontal_indexes.pickle'), 'wb') as f:
	# Pickle the 'data' dictionary using the highest protocol available.
	pickle.dump(indexes, f)

with open(os.path.join(MODELS, 'frontal_sim_indexes.pickle'), 'wb') as f:
	# Pickle the 'data' dictionary using the highest protocol available.
	pickle.dump(sim_indexes, f)

#%%"
#tilted
#for tilted - correspondency between 3D annotations and 2D
model_shape = np.concatenate((tilted_ears, left_eye, tilted_left_nostril, tilted_outline), axis = 0)[:,1:] #ears are not included because they are too variable
indexes = np.vstack(get_indexes(vertices, model_shape))
sim_indexes = indexes.copy()

for j, i in enumerate(indexes):
	if i in zeros:
		sim_indexes[j] = zeros[np.where(zeros == i)[0][0]]

	elif i in correspondence[:,0]:
		sim_indexes[j] = correspondence[np.where(correspondence[:,0] == i)[0][0],1]

	elif i in correspondence[:,1]:
		sim_indexes[j] = correspondence[np.where(correspondence[:,1] == i)[0][0],0]

#sim_indexes = [correspondence[i,1] for i in [np.where(correspondence[:,0] == index)[0] for index in indexes]]

print('len(indexes): ', len(indexes))
print('len(sim indexes): ', len(sim_indexes))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
vert = np.vstack([vertices[i] for i in indexes])
ax.scatter(vert[:, 0], vert[:, 1], vert[:, 2], c = 'r', alpha = 0.1)
vert = np.vstack([vertices[i] for i in sim_indexes])
ax.scatter(vert[:, 0], vert[:, 1], vert[:, 2], c = 'b', alpha = 0.1)
#ax.scatter(model_shape[:, 0], model_shape[:, 1], model_shape[:, 2], c = 'b')
plt.show()


with open(os.path.join(MODELS, 'tilted_indexes.pickle'), 'wb') as f:
	# Pickle the 'data' dictionary using the highest protocol available.
	pickle.dump(indexes, f)

with open(os.path.join(MODELS, 'tilted_sim_indexes.pickle'), 'wb') as f:
	# Pickle the 'data' dictionary using the highest protocol available.
	pickle.dump(sim_indexes, f)
#%%

# profile
left_eye_p = np.asarray([left_eye[i] for i in [3, 4, 5, 0, 1, 2]])
profile_ear = np.asarray([profile_ear[i] for i in [0, 2, 5]])

#cv.imwrite(os.path.join(EX_FOLDER, '%s_BEFORE.png' % (img_path.split('/')[-1].split('.')[0])), img)
#cv.imwrite(os.path.join(EX_FOLDER, '%s_AFTER.png' % (img_path.split('/')[-1].split('.')[0])), img)
model_shape = np.concatenate((profile_ear, cheek, mouth, profile_left_nostril,  left_eye_p, profile_outline[:-7]), axis = 0)[:,1:] #ears are not included because they are too variable
indexes = np.vstack(get_indexes(vertices, model_shape))
sim_indexes = np.vstack([correspondence[i,1] for i in [np.where(correspondence[:,0] == index)[0] for index in indexes]])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
vert = np.vstack([vertices[i] for i in indexes])
ax.scatter(vert[:, 0], vert[:, 1], vert[:, 2], c = 'r', alpha = 0.1)
ax.scatter(model_shape[:, 0], model_shape[:, 1], model_shape[:, 2], c = 'b')
plt.show()

with open(os.path.join(MODELS, 'profile_indexes.pickle'), 'wb') as f:
	# Pickle the 'data' dictionary using the highest protocol available.
	pickle.dump(indexes, f)

with open(os.path.join(MODELS, 'profile_sim_indexes.pickle'), 'wb') as f:
	# Pickle the 'data' dictionary using the highest protocol available.
	pickle.dump(sim_indexes, f)


