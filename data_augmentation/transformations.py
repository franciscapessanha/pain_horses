# %%============================================================================
#                          IMPORTS AND INITIALIZATIONS
# ==============================================================================

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from data_augmentation.ObjLoader import *
import numpy as np
import os
import pickle
import cv2 as cv
from PIL import Image, ExifTags
import math


# %%============================================================================
#                                MAIN
# ==============================================================================
def rotmat_to_angles(R):
	 sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
	 singular = sy < 1e-6
	 if  not singular :
		 x = math.atan2(R[2,1] , R[2,2])
		 y = math.atan2(-R[2,0], sy)
		 z = math.atan2(R[1,0], R[0,0])

	 else :
		 x = math.atan2(-R[1,2], R[1,1])
		 y = math.atan2(-R[2,0], sy)
		 z = 0
	 return np.array([x, y, z])


def fix_angles(r, p, y): # roll, pitch, yaw
	if abs(r) > 90:
		if r > 0:
			r -= 90
		else:
			r += 90
	if p > 90:
		if p > 180:
			p -= 180
		else:
			p -= 90
	elif p < -90:
		if p < -180:
			p += 180
		else:
			p += 90
	return r, p, y


def angles_to_rotmat(yaw, pitch, roll):
		yaw *= np.pi / 180
		pitch *= np.pi / 180
		roll *= np.pi / 180

		a = yaw
		cos_a = np.cos(a)
		sin_a = np.sin(a)
		R_z = np.asarray([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])

		b = pitch
		cos_b = np.cos(b)
		sin_b = np.sin(b)
		R_y = np.asarray([[cos_b, 0, sin_b], [0, 1, 0], [-sin_b, 0, cos_b]])

		y = roll
		cos_y = np.cos(y)
		sin_y = np.sin(y)
		R_x = np.asarray([[1, 0, 0], [0, cos_y, -sin_y], [0, sin_y, cos_y]])

		R_matrix = np.dot(np.dot(R_z, R_y),R_x)
		return R_matrix


# ==============================================================================

def get_angles(rotation_matrix, translation_vector):
	P = np.hstack((rotation_matrix, translation_vector))
	euler_angles_degrees = cv.decomposeProjectionMatrix(P)[6].ravel()
	roll_PnP, pitch_PnP, yaw_PnP = fix_angles(euler_angles_degrees[2],
	                                          euler_angles_degrees[0],
	                                          euler_angles_degrees[1])
	return roll_PnP, pitch_PnP, yaw_PnP
# ==============================================================================

def get_focal_length(img_path):
	img = Image.open(img_path)
	if img._getexif() is not None:
		exif = { ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS }
		if 'FocalLength' in exif:
			print(exif)
			print('focal length: ', exif['FocalLength'])
			focal_length = exif['FocalLength']

			return tuple([focal_length[0], focal_length[0]])
		else:
			size = np.shape(img)[:2]
			print('size IMAGE: ', size)
			return tuple([size[1], size[1]]) # 0.7 w < f < w : crude approximation

	else:
		size = np.shape(img)[:2]
		print('size IMAGE: ', size)
		return tuple([size[1], size[1]]) # 0.7 w < f < w : crude approximation

# ==============================================================================
def get_rigid_transformations(img, img_path, model_shape, img_shape):
	size = img.shape[:2] # height, width, channels

	#print('size image: ', size)
	#focal_length =  get_focal_length(img_path)
	#width = max(size[0], size[1])

	#focal_length = tuple([max(size[0], size[1]), max(size[0], size[1])])
	#focal_length = tuple([focal_length[0], focal_length[0]])
	center = (size[1] / 2, size[0] / 2)
	focal_length = center[0]/np.tan((60/2)* (np.pi/180)) # FOV of about 60 degrees
	focal_length = tuple([focal_length, focal_length])
	# camera calibration matrix (K) - internal camera parameters
	camera_cal_matrix = np.array([[focal_length[0], 0, center[0]], [0, focal_length[1], center[1]], [0, 0, 1]], dtype="double")
	dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

	# Solve Prespective-n-Point problem
	r_vector = np.hstack([0.07190125, 1.26664288, 0.01341055])
	t_vector = np.hstack([106.76842682, 151.01670016, 125.48416302])

	sucess, rotation_vector, translation_vector = cv.solvePnP(np.array(model_shape).reshape((-1, 1, 3)),
	                                                          np.array(img_shape).reshape((-1, 1, 2)),
	                                                          camera_cal_matrix, dist_coeffs,r_vector, t_vector, True,
	                                                          cv.SOLVEPNP_ITERATIVE)

	#print('sucess: ', sucess)
	rotation_matrix = cv.Rodrigues(rotation_vector)[0]
	rotation_matrix = np.asarray(rotation_matrix)
	translation_vector = np.asarray(translation_vector).reshape(-1,1)
	external_parameters = np.concatenate((rotation_matrix, translation_vector), axis = 1) # camera projection matrix

	#camera_matrix = np.dot(np.asarray(camera_cal_matrix), np.asarray(external_parameters))

	euler_angles_degrees = cv.decomposeProjectionMatrix(external_parameters)[6].ravel()


	translation_vector = np.asarray(translation_vector).reshape(-1,1)
	external_parameters = np.concatenate((rotation_matrix, translation_vector), axis = 1) # camera projection matrix

	#camera_matrix = np.dot(np.asarray(camera_cal_matrix), np.asarray(external_parameters))

	euler_angles_degrees = cv.decomposeProjectionMatrix(external_parameters)[6].ravel()

	roll = euler_angles_degrees[2]
	pitch_old =  euler_angles_degrees[0]
	yaw = euler_angles_degrees[1]

	roll, pitch, yaw = fix_angles(roll, pitch_old, yaw)


		#print('rotation vector ', rotation_vector)
		#print('translation vector ', translation_vector)

	rotation_matrix =  angles_to_rotmat(roll, yaw, pitch)
	#print('after: ', rotation_matrix)

	return rotation_matrix, translation_vector, camera_cal_matrix, [roll, pitch, yaw], pitch_old
# ==============================================================================

def project_pts(full_model, K, rotation_matrix = [], translation_vector = [], external_parameters = []):
	projected_model = []
	w = []
	for X in full_model:
		if len(rotation_matrix) == 0:
			if len(external_parameters) == 0:
				x = np.dot(K, X)
				w_x = x #homogenous coordenates
				x = [x[0]/x[2],x[1]/x[2]] # image coordenates
			else:
				camera_matrix = np.dot(np.asarray(K), np.asarray(external_parameters))
				X = np.concatenate((X.reshape(-1,1),np.ones((1,1))), axis = 0)
				x = np.dot(camera_matrix, X)
				w_x = x  #homogeneous coordenates
				x = [x[0]/x[2],x[1]/x[2]] # image coordenates

		else:
			external_parameters = np.hstack((rotation_matrix, translation_vector)) # camera projection matrix
			camera_matrix = np.dot(np.asarray(K), np.asarray(external_parameters))

			X = np.concatenate((X.reshape(-1,1),np.ones((1,1))), axis = 0)
			x = np.dot(camera_matrix, X)
			w_x = x  #homogeneous coordenates
			x = [x[0]/x[2],x[1]/x[2]] # image coordenates

		projected_model.append(np.asarray(x).reshape(1,-1))
		w.append(np.asarray(w_x).reshape(1,-1))

	projected_model = np.vstack(projected_model)
	w = np.vstack(w)
	return projected_model, w

# ==============================================================================

def tps(src_pts, dst_pts, mesh):

	n_points = len(src_pts)
	K = np.zeros((n_points, n_points)) # K - n x n

	for r in range(n_points):
		for c in range(n_points):
			K[r, c] = max(sum((src_pts[r, :] - src_pts[c, :]) ** 2),
			              np.exp(-320))  # avoid having zeros (would give an error in the log)

			K[r, c] = K[r, c] * np.log(K[r, c])
			K[c, r] = K[r, c]

	# P = [1,ps] where ps are n landmark points - 4 x n for 3D
	P = np.concatenate((np.ones((n_points, 1)), src_pts), axis=1)

	# Calculate L matrix
	# L - RBF matrix about source location - (n + 4) x (n + 4)
	first_row = np.concatenate((K, P), axis=1)
	second_row = np.concatenate((P.T, np.zeros((4, 4))), axis=1)
	L = np.concatenate((first_row, second_row), axis=0)  # (n + 3) x (n + 3)

	# Y = L * w
	# Y - Points matrix about destination location
	Y = np.concatenate((dst_pts, np.zeros((4, 3))), axis=0)
	param = np.dot(np.linalg.inv(L), Y)


	# Now that we have the parameters, we can apply then to the full model

	# Now that we have the parameters, we can apply then to the full model
	n_obj_points = len(mesh)
	Kp = np.zeros((n_obj_points, n_points))  # to determine the parameter w U( |P - (x,y)| )

	for r in range(n_obj_points):
		for c in range(n_points):
			Kp[r, c] = max(sum((mesh[r, :] - src_pts[c, :]) ** 2),
			               np.exp(-320))  # avoid having zeros (would give an error in the log)!
			Kp[r, c] = Kp[r, c] * np.log(Kp[r, c])

	P = np.concatenate((np.ones((n_obj_points, 1)), mesh), axis=1)
	L = np.concatenate((Kp, P), axis=1)
	new_points = np.dot(L, param)

	return new_points