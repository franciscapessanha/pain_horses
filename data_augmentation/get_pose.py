
# %%============================================================================
#                          IMPORTS AND INITIALIZATIONS
# ==============================================================================

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from ObjLoader import *
import numpy as np
import os
import pickle
import cv2 as cv
from mpl_toolkits.mplot3d import Axes3D
import glob
from load_obj import *
from utils import *
from transformations import *

DATASET = os.path.join(os.getcwd(), '..', 'dataset')
MODELS =  os.path.join(DATASET, '3D_annotations', 'models')
LANDMARKS =  os.path.join(DATASET, '3D_annotations', 'landmarks')
ANGLES =  os.path.join(DATASET, '3D_annotations', 'angles')

PLOTS = os.path.join(os.getcwd(), 'plots')
if not os.path.exists(PLOTS):
	os.mkdir(PLOTS)


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




data = pickle.load(open(os.path.join(DATASET, 'lms_annotations.pkl'), "rb"))

FRONTAL_FOLDER = os.path.join(os.getcwd(), 'gt_pose_frontal')
TILTED_FOLDER = os.path.join(os.getcwd(), 'gt_pose_tilted')
PROFILE_FOLDER = os.path.join(os.getcwd(), 'gt_pose_profile')

if not os.path.exists(FRONTAL_FOLDER):
	os.mkdir(FRONTAL_FOLDER)

if not os.path.exists(PROFILE_FOLDER):
	os.mkdir(PROFILE_FOLDER)

if not os.path.exists(TILTED_FOLDER):
	os.mkdir(TILTED_FOLDER)

def rotate_points(points):
	R_z = 0
	R_y = 0
	R_x = 90

	R_matrix = angles_to_rotmat(R_z, R_y, R_x)
	points_rot = []
	for pt in points:
	    points_rot.append(np.dot(R_matrix, pt))
	points_rot = np.vstack(points_rot)
	return points_rot

# %%============================================================================
#                                MAIN
# ==============================================================================

vertices, triangles = load_obj(HEAD_OBJ)
correspondence, zeros =  symetric(vertices)

vertices = rotate_points(vertices)
center = np.mean(vertices, axis = 0)

tilted_outline = rotate_points(load_lms_model(TILTED_OUTLINE)[:,1:])
tilted_ears = rotate_points(load_lms_model(TILTED_EARS)[:,1:])
tilted_left_nostril = rotate_points(load_lms_model(TILTED_LEFT_NOSTRIL)[:,1:])

left_ear = rotate_points(load_lms_model(FRONTAL_LEFT_EAR)[:,1:])
frontal_outline = rotate_points(load_lms_model(FRONTAL_OUTLINE)[:,1:])
frontal_left_nostril = rotate_points(load_lms_model(FRONTAL_LEFT_NOSTRIL)[:,1:])

profile_ear = rotate_points(load_lms_model(PROFILE_LEFT_EAR)[:,1:])
profile_outline = rotate_points(load_lms_model(PROFILE_OUTLINE)[:,1:])
profile_left_nostril = rotate_points(load_lms_model(PROFILE_LEFT_NOSTRIL)[:,1:])

cheek = rotate_points(load_lms_model(CHEEK)[:,1:])
mouth = rotate_points(load_lms_model(MOUTH)[:,1:])

left_eye = rotate_points(load_lms_model(LEFT_EYE)[:,1:])


#%%

all_profile = [i for i in data.values if (i[2] == -60 or i[2] == 60) and i[1] == 'horse']

all_tilted = [i for i in data.values if (i[2] == -30 or i[2] == 30) and i[1] == 'horse']

all_frontal = [i for i in data.values if (i[2] == 0) and i[1] == 'horse']

#%%
def get_pose_tilted(all_tilted):
	pose_info_tilted = []
	for img_info in all_tilted:
		pose = img_info[2]
		lms = img_info[4]
		img_path = os.path.join(os.getcwd(), '..', img_info[0])
		img = cv.imread(img_path)

		img, lms = flip_image(img, lms, pose)
		ears = [tilted_ears[i] for i in [0,2,3,5]]
		#for tilted - correspondency between 3D annotations and 2D
		model_shape = np.concatenate((ears, left_eye, tilted_left_nostril, tilted_outline), axis = 0) #ears are not included because they are too variable
		img_shape = np.vstack([lms[i] for i in [0, 4, 5, 9, *range(10,33)]])


		rotation_matrix, translation_vector, K, angles = get_rigid_transformations(img, img_path, model_shape, img_shape)

		roll, pitch, yaw = angles

		if abs(yaw) != 0:
			proj_pts, _ = project_pts(vertices, K, rotation_matrix, translation_vector)
			"""
			for pt in proj_pts:
				cv.circle(img,tuple((int(pt[0]), int(pt[1]))), 3, (255,255,255), -1) #gt lms = blue
			"""
			rotation = angles_to_rotmat(0, 0, 0)
			points = np.float32([center + [20, 0, 0], center + [0, 20, 0], center + [0, 0, -20], center]).reshape(-1, 3)
			axisPoints, _ = project_pts(points, K, rotation_matrix, translation_vector)
			axisPoints = axisPoints.astype(int)
			cv.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (0,0,255), 10)
			cv.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 10)
			cv.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (255,0,0), 10)
			cv.imwrite(os.path.join(TILTED_FOLDER, 'yaw_%d_pitch_%d_roll_%d_%s.png' % (int(yaw), int(pitch), int(roll), img_path.split('/')[-1].split('.')[0])), img)

		pose_info_tilted.append([img_info[0].split('/')[-1],roll, pitch, np.sign(pose) * yaw]) # path, roll, pitch, yaw

	pose_info_tilted = np.vstack(pose_info_tilted)
	return pose_info_tilted

def get_pose_frontal(all_frontal):
	pose_info_frontal = []
	for img_info in all_frontal:
		pose = img_info[2]
		lms = img_info[4]
		img_path = os.path.join(os.getcwd(), '..', img_info[0])
		img = cv.imread(img_path)

		img, lms = flip_image(img, lms, pose)

		#for tilted - correspondency between 3D annotations and 2D
		right_ear = [[-pt[1], pt[2], pt[3]] for pt in load_lms_model(FRONTAL_LEFT_EAR)]
		right_ear = rotate_points([right_ear[i] for i in range(len(right_ear)-1, -1, -1)])
		right_eye = [[-pt[1], pt[2], pt[3]] for pt in load_lms_model(LEFT_EYE)]
		right_eye = rotate_points([right_eye[i] for i in [2, 1, 0, 5, 4, 3]])
		right_nostril = [[-pt[1], pt[2], pt[3]] for pt in load_lms_model(FRONTAL_LEFT_NOSTRIL)]
		right_nostril = rotate_points([right_nostril[i] for i in [2, 1, 0, 5, 4, 3]])

		model_shape = np.concatenate((left_eye, frontal_left_nostril, right_nostril, right_eye, frontal_outline), axis = 0) #ears are not included because they are too variable
		img_shape = np.vstack([lms[i] for i in [*range(10,35), *range(43,46)]])

		rotation_matrix, translation_vector, K, angles = get_rigid_transformations(img,img_path, model_shape, img_shape)

		roll, pitch, yaw = angles

		if abs(yaw) != 0:
			proj_pts, _ = project_pts(vertices, K, rotation_matrix, translation_vector)
			"""
			for pt in proj_pts:
				cv.circle(img,tuple((int(pt[0]), int(pt[1]))), 3, (255,255,255), -1) #gt lms = blue
			"""

			rotation = angles_to_rotmat(0, 0, 0)
			points = np.float32([center + [20, 0, 0], center + [0, 20, 0], center + [0, 0, -20], center]).reshape(-1, 3)
			axisPoints, _ = project_pts(points, K, rotation_matrix, translation_vector)
			axisPoints = axisPoints.astype(int)
			cv.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (0,0,255), 10)
			cv.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 10)
			cv.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (255,0,0), 10)
			cv.imwrite(os.path.join(FRONTAL_FOLDER, 'yaw_%d_pitch_%d_roll_%d_%s.png' % (int(yaw), int(pitch), int(roll), img_path.split('/')[-1].split('.')[0])), img)

		pose_info_frontal.append([img_info[0].split('/')[-1],roll, pitch, yaw]) # path, roll, pitch, yaw

	pose_info_frontal = np.vstack(pose_info_frontal)

	return pose_info_frontal
#%%
def get_pose_profile(all_profile):
	pose_info_profile = []
	for img_info in all_profile[173:]:
		pose = img_info[2]
		lms = img_info[4]
		img_path = os.path.join(os.getcwd(), '..', img_info[0])
		img = cv.imread(img_path)
		left_eye_p = [left_eye[i] for i in [3, 4, 5, 0, 1, 2]]
		#cv.imwrite(os.path.join(EX_FOLDER, '%s_BEFORE.png' % (img_path.split('/')[-1].split('.')[0])), img)
		img, lms = flip_image(img, lms, pose)
		#cv.imwrite(os.path.join(EX_FOLDER, '%s_AFTER.png' % (img_path.split('/')[-1].split('.')[0])), img)
		model_shape = np.concatenate(([profile_ear[0,:]], [profile_ear[-1,:]], cheek, mouth, profile_left_nostril,  left_eye_p, profile_outline[:-7], [profile_outline[-2,:]], [profile_outline[-1,:]]), axis = 0) #ears are not included because they are too variable
		img_shape = np.vstack([lms[i] for i in [0, 5, *range(6,10),*range(12,38), 43, 44]])


		rotation_matrix, translation_vector, K, angles = get_rigid_transformations(img, img_path, model_shape, img_shape)

		proj_lms, _ = project_pts(model_shape, K, rotation_matrix, translation_vector)

		roll, pitch, yaw = angles
		#yaw = yaw * np.sign(pose)

		if abs(yaw) != 0:
			proj_pts, _ = project_pts(vertices, K, rotation_matrix, translation_vector)
			"""
			for pt in proj_pts:
				cv.circle(img,tuple((int(pt[0]), int(pt[1]))), 3, (255,255,255), -1) #gt lms = blue
			"""
			rotation = angles_to_rotmat(0, 0, 0)
			points = np.float32([center + [20, 0, 0], center + [0, 20, 0], center + [0, 0, -20], center]).reshape(-1, 3)
			axisPoints, _ = project_pts(points, K, rotation_matrix, translation_vector)
			axisPoints = axisPoints.astype(int)
			cv.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (0,0,255), 10)
			cv.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 10)
			cv.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (255,0,0), 10)
			cv.imwrite(os.path.join(PROFILE_FOLDER, 'yaw_%d_pitch_%d_roll_%d_%s.png' % (int(yaw), int(pitch), int(roll), img_path.split('/')[-1].split('.')[0])), img)

		pose_info_profile.append([img_info[0].split('/')[-1],roll, pitch, np.sign(pose) * yaw]) # path, roll, pitch, yaw


	return pose_info_profile

#%%
#pose_info_tilted = get_pose_tilted(all_tilted)
pose_info_tilted = np.load(open(os.path.join(ANGLES,'tilted_roll_pitch_yaw.pickle'), 'rb'), allow_pickle = True)
checked_tilted = glob.glob(os.path.join(TILTED_FOLDER, '*.png'))
list_indexes = [int(i.split('/')[-1].split('_')[-1].split('.')[0]) for i in checked_tilted]
pose_info_tilted  = [i for i in pose_info_tilted if int(i[0].split('.')[0]) in list_indexes]
pose_info_tilted = np.vstack(pose_info_tilted)

# TILTED
#=========
with open(os.path.join(ANGLES,'tilted_roll_pitch_yaw.pickle'), 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(pose_info_tilted, f)


plt.figure()
roll_tilted_counts, roll_tilted_bins, _ = plt.hist(pose_info_tilted[:,1].astype(np.float), bins=36, range=(-90,90))
with open(os.path.join(ANGLES,'roll_tilted_counts_vs_bins.pickle'), 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump([roll_tilted_counts, roll_tilted_bins], f)
title = 'Roll distribution - tilted pose'
plt.title(title)
plt.xlabel('angle (degrees)')
plt.ylabel('number of occurrences')
plt.savefig(os.path.join(PLOTS, title))

plt.figure()
pitch_tilted_counts, pitch_tilted_bins, _ = plt.hist(pose_info_tilted[:,2].astype(np.float), bins=36, range=(-90,90))
with open(os.path.join(ANGLES,'pitch_tilted_counts_vs_bins.pickle'), 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump([pitch_tilted_counts, pitch_tilted_bins], f)
title = 'Pitch distribution - tilted pose'
plt.title(title)
plt.xlabel('angle (degrees)')
plt.ylabel('number of occurrences')
plt.savefig(os.path.join(PLOTS, title))

plt.figure()
yaw_tilted_counts, yaw_tilted_bins, _ = plt.hist(pose_info_tilted[:,3].astype(np.float), bins=36, range=(-90,90))
with open(os.path.join(ANGLES,'yaw_tilted_counts_vs_bins.pickle'), 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump([yaw_tilted_counts, yaw_tilted_bins], f)
title = 'Yaw distribution - tilted pose'
plt.title(title)
plt.xlabel('angle (degrees)')
plt.ylabel('number of occurrences')
plt.savefig(os.path.join(PLOTS, title))


#%%
# PROFILE
#=========
#pose_info_profile = get_pose_profile(all_profile)
checked_profile = glob.glob(os.path.join(PROFILE_FOLDER, '*.png'))
list_indexes = [int(i.split('/')[-1].split('_')[-1].split('.')[0])for i in checked_profile]
pose_info_profile  = [i for i in pose_info_profile if int(i[0].split('.')[0]) in list_indexes]
pose_info_profile = np.vstack(pose_info_profile)

with open(os.path.join(ANGLES,'profile_roll_pitch_yaw.pickle'), 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(pose_info_profile, f)


plt.figure()
roll_profile_counts, roll_profile_bins, _ = plt.hist(pose_info_profile[:,1].astype(np.float), bins=36, range=(-90,90))
with open(os.path.join(ANGLES,'roll_profile_counts_vs_bins.pickle'), 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump([roll_profile_counts, roll_profile_bins], f)
title = 'Roll distribution - profile pose'
plt.title(title)
plt.xlabel('angle (degrees)')
plt.ylabel('number of occurrences')
plt.savefig(os.path.join(PLOTS, title))

plt.figure()
pitch_profile_counts, pitch_profile_bins, _ = plt.hist(pose_info_profile[:,2].astype(np.float), bins=36, range=(-90,90))
with open(os.path.join(ANGLES,'pitch_profile_counts_vs_bins.pickle'),'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump([pitch_profile_counts, pitch_profile_bins], f)
title = 'Pitch distribution - profile pose'
plt.title(title)
plt.xlabel('angle (degrees)')
plt.ylabel('number of occurrences')
plt.savefig(os.path.join(PLOTS, title))

plt.figure()
yaw_profile_counts, yaw_profile_bins, _ = plt.hist(pose_info_profile[:,3].astype(np.float), bins=36, range=(-90,90))
with open(os.path.join(ANGLES,'yaw_profile_counts_vs_bins.pickle'), 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump([yaw_profile_counts, yaw_profile_bins], f)
title = 'Yaw distribution - profile pose'
plt.title(title)
plt.xlabel('angle (degrees)')
plt.ylabel('number of occurrences')
plt.savefig(os.path.join(PLOTS, title))


#%%
 # FRONTAL
#=========
#pose_info_frontal = get_pose_frontal(all_frontal)
checked_frontal = glob.glob(os.path.join(FRONTAL_FOLDER, '*.png'))
list_indexes = [int(i.split('/')[-1].split('_')[-1].split('.')[0]) for i in checked_frontal]
pose_info_frontal  = [i for i in pose_info_frontal if int(i[0].split('.')[0]) in list_indexes]
pose_info_frontal = np.vstack(pose_info_frontal)


with open(os.path.join(ANGLES,'frontal_roll_pitch_yaw.pickle'), 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(pose_info_frontal, f)

plt.figure()
roll_frontal_counts, roll_frontal_bins, _ = plt.hist(pose_info_frontal[:,1].astype(np.float), bins=36, range=(-90,90))
with open(os.path.join(ANGLES,'roll_frontal_counts_vs_bins.pickle'), 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump([roll_frontal_counts, roll_frontal_bins], f)
title = 'Roll distribution - frontal pose'
plt.title(title)
plt.xlabel('angle (degrees)')
plt.ylabel('number of occurrences')
plt.savefig(os.path.join(PLOTS, title))

plt.figure()
pitch_frontal_counts, pitch_frontal_bins, _ = plt.hist(pose_info_frontal[:,2].astype(np.float), bins=36, range=(-90,90))
with open(os.path.join(ANGLES,'pitch_frontal_counts_vs_bins.pickle'), 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump([pitch_frontal_counts, pitch_frontal_bins], f)
title = 'Pitch distribution - frontal pose'
plt.title(title)
plt.xlabel('angle (degrees)')
plt.ylabel('number of occurrences')
plt.savefig(os.path.join(PLOTS, title))

plt.figure()
yaw_frontal_counts, yaw_frontal_bins, _ = plt.hist(pose_info_frontal[:,3].astype(np.float), bins=36, range=(-90,90))
with open(os.path.join(ANGLES,'yaw_frontal_counts_vs_bins.pickle'), 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump([yaw_frontal_counts, yaw_frontal_bins], f)
title = 'Yaw distribution - frontal pose'
plt.title(title)
plt.xlabel('angle (degrees)')
plt.ylabel('number of occurrences')
plt.savefig(os.path.join(PLOTS, title))
