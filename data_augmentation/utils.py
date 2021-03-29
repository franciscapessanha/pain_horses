# %%============================================================================
#                          IMPORTS AND INITIALIZATIONS
# ==============================================================================

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from data_augmentation.ObjLoader import *
import numpy as np
import os
import pickle
import cv2 as cv
from data_augmentation.load_obj import *
from random import randint
from data_augmentation.transformations import *
from pylab import *
import scipy.linalg as lin
#from pygem import RBFParameters, RBF, IDWParameters, IDW
import trimesh
import utils
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

EX_FOLDER = os.path.join(os.getcwd(), 'examples')

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

"""
Fix Angles
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Fix relative to 3D coordenates

Arguments:
		* r: roll (x)
		* p: pitch (y)
		* y: yaw (z)
Return:
		* r, p, y altered
---------------------------------------------------------------------------
"""
def fix_angles(r, p, y): #convert angles to the "conventional axis system"
	"""
	yaw = p #rotation according to z has a diferent direction
	roll = y
	pitch = r - 45
	"""
	if abs(r) > 90:
		if r > 0:
			r -= 180
		else:
			r += 180
	if p > 0:
		p -= 100
	else:
		p += 100

	return r, p, y



def ckeck_point_correspondence(img, img_shape, model_shape):
	n_points = len(model_shape)
	colors_bgr = []
	colors_rgb = []

	for i in range(n_points):
		b = randint(0,255)
		g = randint(0,255)
		r = randint(0,255)

		colors_bgr.append(tuple((b, g, r)))
		colors_rgb.append(tuple((r, g, b)))

	for i, pt in enumerate(img_shape):
		cv.circle(img,tuple((int(pt[0]), int(pt[1]))), 3, colors_bgr[i], -1)
	cv.imwrite(os.path.join(EX_FOLDER, img_path.split('/')[-1]), img)


def save_plot(points, img_path, extension, c = 'r'):
 	fig = plt.figure()
 	ax = fig.add_subplot(111, projection='3d')
 	ax.scatter(points[:, 0], points[:, 1], points[:, 2], c = c)
 	plt.title(img_path.split('/')[-1])
 	plt.savefig(os.path.join(EX_FOLDER, img_path.split('/')[-1].split('.')[0] + '_%s.png' % extension))
 	plt.show()

def save_2D_plot(points, img_path, extension, c = 'r'):
	fig = plt.figure()
	for i,pt in enumerate(points):
		plt.scatter(pt[0], pt[1], c = c)
		annotate(str(i), pt)
	plt.title(img_path.split('/')[-1])
	plt.savefig(os.path.join(EX_FOLDER, img_path.split('/')[-1].split('.')[0] + '_%s.png' % extension))
	plt.show()


# ==============================================================================
def save_img_pts(pts,img_shape, img, img_path, indexes, extension):
	ind_lms = [int(i) for i in indexes]
	lms= np.asarray([pts[i] for i in ind_lms])


	for pt in img_shape:
		cv.circle(img,tuple((int(pt[0]), int(pt[1]))), 4, (0,0,255), -1)

	for pt in lms:
		cv.circle(img,tuple((int(pt[0]), int(pt[1]))), 4, (255,0,0), -1)


	for pt in pts:
		cv.circle(img,tuple((int(pt[0]), int(pt[1]))), 2, (255,255,255), -1)

	cv.imwrite(os.path.join(EX_FOLDER, img_path.split('/')[-1].split('.')[0] + '_%s.jpg' % extension), img)

# ==============================================================================
def get_back_projection(pt, K):
		inv_K = np.linalg.inv(K)
		X =  np.dot(inv_K, pt)
		X = np.squeeze(np.asarray([X[0], X[1], X[2]]))

		return X
# ==============================================================================
"""
def get_recomputed_vertice(vertice, w):
		dist = np.sqrt((init_vertice[0] - init_back_proj[0])**2
 								 + (init_vertice[1] - init_back_proj[1])**2
 								 + (init_vertice[2] - init_back_proj[2])**2)

		normal_vector = np.asarray([w[0,0], w[1,0], -1])
		unit_vector = normal_vector/np.linalg.norm(normal_vector)


		#this dist will be constant since the movements will be made according to the same plane
		update_vertice = proj_pt + dist*np.sign(np.dot((init_vertice - init_back_proj), unit_vector))*unit_vector

		#update_vertice = np.asarray([update_vertice[0], update_vertice[1], init_vertice[-1]])
		return update_vertice
"""
# ==============================================================================
def get_simetric_points(simetry_plan, points, excluded = [-1]):

	A = np.concatenate((simetry_plan[:,0].reshape(-1,1), simetry_plan[:,1].reshape(-1,1), np.ones((len(simetry_plan[:,1]),1))), axis = 1)
	B = simetry_plan[:,2]
	ans = np.dot(A.T,A)
	ans = np.linalg.inv(ans)
	ans = np.dot(ans,A.T)
	w = np.dot(ans,B)

	normal_vector = np.asarray([w[0], w[1], -1])
	unit_vector = normal_vector/np.linalg.norm(normal_vector)

	#considering origin = [0,0,0]
	origin = np.asarray([0,0,w[-1]])
	simetric = np.zeros((np.shape(points)[0] * 2, 3))

	for i,pt in enumerate(points):
		v = pt - origin
		dist = np.dot(v,unit_vector)
		projected_point = pt - dist*normal_vector

		dist = np.sqrt((projected_point[0] - pt[0])**2
 								 + (projected_point[1] - pt[1])**2
 								 + (projected_point[2] - pt[2])**2)

		if i in range(33, 44):
			#print('dist: ', dist)
			if dist > 5:
				excluded.append([i, i + np.shape(points)[0]])

		if np.sign(np.dot((pt - projected_point), unit_vector)) == -1:
			simetric[i,:] = projected_point - dist*np.sign(np.dot((pt - projected_point), unit_vector))*unit_vector
			simetric[i + np.shape(points)[0],:] = pt
		else:
			simetric[i + np.shape(points)[0],:] = projected_point - dist*np.sign(np.dot((pt - projected_point), unit_vector))*unit_vector
			simetric[i,:] = pt

	if len(excluded) > 0:
		excluded = np.hstack(excluded)
	simetric = np.vstack([simetric[i] for i in range(len(simetric)) if i not in excluded])
	return simetric, excluded


# ==============================================================================
def symetric(vertices):
	correspondence = []
	zeros = []
	for i, pt in enumerate(vertices):
		if pt[0] > 0: #x > 0
			j = np.where((np.round(vertices,4) == np.round([-pt[0], pt[1], pt[2]],4)).all(axis = 1))[0][0]
			correspondence.append([i,j])
		elif pt[0] == 0:
				zeros.append(i)

	correspondence = np.vstack(correspondence)
	return correspondence, zeros

# ==============================================================================
"""
def rgb_to_hsl(color):
	r = color[0]/255
	g = color[1]/255
	b = color[2]/255

	min_ = min([r,g,b])
	max_ = max([r,g,b])

	l = (min_ + max_)/2

	if l < 0.5:
		s = (max_ - min_) / (max_ + min_)
	else:
		s = (max_ - min_) / (2.0 - max_ - min_)

	if max_ == r:
		h = (g - b) / (max_ - min_)

	elif max_ == g:
		h = 2.0 + (b - r) / (max_ - min_)

	elif max_ == b:
		h = 4.0 + (r - g) / (max_ - min_)

	return [h,s,v]
"""
# ==============================================================================
def crop_image(img, img_path, lms, pose):
	if pose > 0:
		img = cv.flip(img, 1)
		h, w = np.shape(img)[:2]
		mirror_lms = []
		for pt in lms:
			new_pt = [w - pt[0], pt[1]]
			mirror_lms.append(new_pt)

		lms = np.vstack(mirror_lms)
	error = 0.05
	lms = np.vstack(lms)
	lms_x = lms[:,0]
	lms_y = lms[:,1]

	img_h, img_w = img.shape[:2]
	x_min =  max(0,int(min(lms_x) - error * img_w))
	x_max = min(img_w, int(max(lms_x) + error * img_w))

	y_min = max(0, int(min(lms_y) - error * img_h))
	y_max = min(img_h, int(max(lms_y) + error * img_h))

	img_crop = img[y_min : y_max, x_min : x_max]
	crop_h, crop_w = img_crop.shape[:2]

	if crop_h >  crop_w:
		new_h = 855
		new_w = int((crop_w * new_h) / crop_h)

	elif crop_w >= crop_h:
		new_w = 855
		new_h = int((crop_h * new_w) / crop_w)


	img_resize = cv.resize(img_crop, (new_w, new_h))
	lms_resize = []
	for pt in lms:
		new_pt = ((pt[0] - x_min) * new_w/crop_w, (pt[1] - y_min) * new_h/crop_h)
		#new_pt = ((pt[0] - x_min), (pt[1] - y_min))
		lms_resize.append(new_pt)


	lms_resize = np.vstack(lms_resize)


	#cv.imwrite(os.path.join(EX_FOLDER, img_path.split('/')[-1].split('.')[0] + '_%s.jpg' % 'gt_resized'), img_resize)


	return img_resize, lms_resize


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



def get_tps_back_projection(img, img_path, model_shape, img_shape, vertices, indexes, zeros, correspondence):
	rotation_matrix, translation_vector, K = get_rigid_transformations(img, img_path, model_shape, img_shape)
	# homogenous_pts - homogenous coordinates of the projected points
	_, homogenous_pts = project_pts(vertices, K, rotation_matrix, translation_vector)


	#projected_ears, homo_ear_pts = project_pts(vert_ears, K, rotation_matrix, translation_vector)

	#save_img_pts(projected_model, img, img_path, indexes, 'rigid')

	back_projection = np.zeros(np.shape(vertices))
	for i in range(len(vertices)):
		back_projection[i][:] = get_back_projection(homogenous_pts[i,:], K)

	ind_lms = [int(i) for i in indexes]
	lms_back_projection= np.asarray([back_projection[i,:] for i in ind_lms])
	lms_homo_pts = np.asarray([homogenous_pts[i,:] for i in ind_lms])
# ear_back_projection = np.zeros(np.shape(vert_ears))
# 	for i in range(len(vert_ears)):
# 		ear_back_projection[i][:] = get_back_projection(homo_ear_pts[i,:], K)

	#1. back projection
	img_lms_bp = np.zeros((len(img_shape), 3))
	for i in range(len(img_lms_bp)):
			img_lms_bp[i][:] = get_back_projection(np.asarray([img_shape[i,0]*lms_homo_pts[i,-1],
																														img_shape[i,1]*lms_homo_pts[i,-1],
																														lms_homo_pts[i,-1]]), K) # assuming w is the depth


	simetry_plan = np.vstack([back_projection[i] for i in zeros]) # the simetry plan will be the plan of unique points

	# GET TPS
	# ==============
	print('dim lms_back_projection: ',  np.shape(lms_back_projection))
	print('dim img_lms_bp: ',  np.shape(img_lms_bp))
	print('dim back_projection: ',  np.shape(back_projection))
	tps_back_projection = tps(lms_back_projection, img_lms_bp, back_projection)

	#Reproject all points
	pts_all, _ = project_pts(tps_back_projection, K)
	#save_img_pts(pts_all, img_shape, img, img_path, indexes, 'pre_symetry') # works ok!

	# get simetric tps
	tps_back_projection_neg = np.vstack([tps_back_projection[i] for i in correspondence[:,0]]) #the image landmarks are on the negative side
	tps_back_projection_zeros = np.vstack([tps_back_projection[i] for i in zeros])
	tps_back_projection_pos =  get_simetric_points(simetry_plan, tps_back_projection_neg)

	indexes_p = correspondence[:,1]
	indexes_n = correspondence[:,0]

	tps_back_projection = []
	for index in range(len(vertices)):
		ind_zero =  np.where(np.asarray(zeros) == index)
		ind_pos =  np.where(indexes_p == index)
		ind_neg =  np.where(indexes_n == index)

		if len(ind_zero[0]) != 0:
			tps_back_projection.append(tps_back_projection_zeros[ind_zero[0]])
		elif len(ind_pos[0]) != 0:
			tps_back_projection.append(tps_back_projection_pos[ind_pos[0]])
		elif len(ind_neg[0]) != 0:
			tps_back_projection.append(tps_back_projection_neg[ind_neg[0]])

	tps_back_projection = np.vstack(tps_back_projection)

	return K, tps_back_projection