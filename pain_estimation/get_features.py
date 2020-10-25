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
from utils import *
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
from skimage import exposure
import random
import cv2



BIG_SIDE = 128

EX_FOLDER = os.path.join(os.getcwd(), 'pain_estimation', 'examples')

if not os.path.exists(EX_FOLDER):
	os.mkdir(EX_FOLDER)
# ==============================================================================

def update_landmarks(lms, pose):
	if abs(pose) == 0:
		img_shape = np.vstack([lms[i] for i in [0, 2, 4, 5, 7, 9, range(10,35), *range(43,46)]])
	elif abs(pose) == 30:
		img_shape = np.vstack([lms[i] for i in [0, 2, 4, 5, 7, 9, *range(10,33)]])
	elif abs(pose) == 60:
		img_shape = np.vstack([lms[i] for i in [0, 2, 5, *range(6,10),*range(12,38)]])

	return img_shape

# ==============================================================================
def resize_img(img,ratio):
	init_h, init_w = np.shape(img)[:2]
	if ratio < 1:
		width = BIG_SIDE
		height = BIG_SIDE * ratio
	else:
		height = BIG_SIDE
		width = BIG_SIDE / ratio

	img_resize = cv.resize(img, (int(width), int(height)))

	return img_resize
# ==============================================================================
def rotate_image(label, image, points, angle):
	h, w = image.shape[:2]
	img_c = (w / 2, h / 2)

	rot = cv.getRotationMatrix2D(img_c, angle, 1)

	rad = math.radians(angle)
	sin = math.sin(rad)
	cos = math.cos(rad)
	b_w = int((h * abs(sin)) + (w * abs(cos)))
	b_h = int((h * abs(cos)) + (w * abs(sin)))

	rot[0, 2] += ((b_w / 2) - img_c[0])
	rot[1, 2] += ((b_h / 2) - img_c[1])

	new_image= cv.warpAffine(image, rot, (b_w, b_h), flags=cv.INTER_LINEAR)
	n_h, n_w, _ = np.shape(new_image)
	ones = np.ones(shape=(len(points), 1))
	points_ones = np.hstack([points, ones])
	#new_points = cv.transform(points_ones, M)
	new_points = rot.dot(points_ones.T).T
	min_x = min(new_points[:, 0])
	max_x = max(new_points[:, 0])
	min_y = min(new_points[:, 1])
	max_y = max(new_points[:, 1])
	bw = max_x - min_x
	bh = max(max_y - min_y, 0.05*h)

	#print('min x: ', min_x, 'max x: ', max_x, 'min_y: ', min_y, 'max_y: ', max_y)
	min_x = max(0, int(min_x - 0.10 * bw))
	max_x = min(int(max_x + 0.10 * bw),n_w)
	min_y = max(0, int(min_y - 0.10 * bh))
	max_y = min(int(max_y + 0.10 * bh),n_h)

	#print('min x: ', min_x, 'max x: ', max_x, 'min_y: ', min_y, 'max_y: ', max_y)
	#print('len image: ', np.shape(new_image))
	#print('initial lms: ', points)
	crop_img = new_image[min_y:max_y, min_x:max_x]

	n_h, n_w = crop_img.shape[:2]
	if n_w == 0:
		return [], new_points
	ratio = n_h/n_w
	crop_img = resize_img(crop_img, ratio)

	#print('len crop image: ', np.shape(crop_img))

	return crop_img, new_points

# ==============================================================================
def rotate_horizontaly(label, img, points, base, top):

	if top[0] != base[0] and top[1] != base[1]:
		angle = np.arctan((base[1] - top[1])/(base[0] - top[0])) #radians
		if top[0] > base[0]: #rotation clockwise
			angle = angle * 180/math.pi

		else: #anticlockwise
			angle = - angle * 180/math.pi
			angle = 360 - angle

	else:
		angle = 0


	n_img, n_points = rotate_image(label, img, points, angle)

	if len(n_img) == 0:
		angle = 0

	return n_img, angle
# ==============================================================================

def rotate_vertically(label, img, points, base, top):

	if top[0] != base[0] and top[1] != base[1]:
		angle = np.arctan((base[1] - top[1])/(base[0] - top[0])) #radians
		if base[0] < top[0]: #rotation clockwise
			angle = 90 + angle * 180/math.pi
		else: #anticlockwise
			angle = 90 - angle * 180/math.pi
			angle = 360 - angle

	else:
		angle = 0


	n_img, n_points = rotate_image(label, img, points, angle)

	if len(n_img) == 0:
		angle = 0
	return n_img, angle

# ==============================================================================
def get_ears(img, lms, pose):
	label = 'ear_%d' % pose
	if abs(pose) == 0:
		r_ear = [lms[i] for i in [0, 1, 2]]
		base = np.mean([r_ear[0], r_ear[-1]], axis = 0)
		top = r_ear[1]
		r_ear, r_rot = rotate_vertically(label, img, r_ear, base, top)
		l_ear = [lms[i] for i in [3, 4, 5]]
		base = np.mean([l_ear[0], l_ear[-1]], axis = 0)
		top = l_ear[1]
		l_ear, l_rot = rotate_vertically(label,img, l_ear, base, top)

	elif abs(pose) == 30:

		r_ear = [lms[i] for i in [0, 1, 2]]
		base = np.mean([r_ear[0], r_ear[-1]], axis = 0)
		top = r_ear[1]
		r_ear, r_rot = rotate_vertically(label,img, r_ear, base, top)

		l_ear = [lms[i] for i in [3, 4, 5]]
		base = np.mean([l_ear[0], l_ear[-1]], axis = 0)
		top = l_ear[1]
		l_ear, l_rot =  rotate_vertically(label, img, l_ear, base, top)


	elif abs(pose) == 60:
		l_ear = [lms[i] for i in [0, 1, 2]]
		base = np.mean([l_ear[0], l_ear[-1]], axis = 0)
		top = l_ear[1]
		l_ear, l_rot =  rotate_vertically(label, img,l_ear, base, top)
		r_ear = []
		r_rot = 0

	return [l_ear, r_ear], [l_rot, r_rot]

# ==============================================================================

def get_eyes(img, lms, pose):
	label = 'eye_%d' % pose
	if abs(pose) == 0:
		l_eye = [lms[i] for i in [*range(6,12)]]
		base = l_eye[-3]
		top = l_eye[0]
		l_eye, l_rot = rotate_horizontaly(label, img, l_eye, base, top)

		r_eye = [lms[i] for i in [*range(24,30)]]
		base = r_eye[-1]
		top = r_eye[3]
		r_eye, r_rot = rotate_horizontaly(label, img, r_eye, base, top)

	elif abs(pose) == 30:
		l_eye = [lms[i] for i in [*range(6,12)]]
		base = l_eye[-3]
		top = l_eye[0]
		l_eye, l_rot = rotate_horizontaly(label, img, l_eye, base, top)
		r_eye = []
		r_rot = 0

	elif abs(pose) == 60:
		l_eye = [lms[i] for i in [*range(17,23)]]
		base = l_eye[0]
		top = l_eye[-3]
		l_eye, l_rot = rotate_horizontaly(label, img,l_eye, base, top)
		r_eye = []
		r_rot = 0
	return [l_eye, r_eye], [l_rot, r_rot]

# ==============================================================================
def get_nostrils(img, lms, pose):
	label = 'nose_%d' % pose
	if abs(pose) == 0:
		l_nostril = [lms[i] for i in [*range(12,18)]]
		base = l_nostril[2]
		top = l_nostril[0]
		l_nostril, l_rot = rotate_vertically(label, img, l_nostril, base, top)

		r_nostril = [lms[i] for i in [*range(18,24)]]
		base = r_nostril[0]
		top = r_nostril[2]
		r_nostril, r_rot = rotate_vertically(label, img, r_nostril, base, top)
		r_nostril = []
		r_rot = 0

	elif abs(pose) == 30:
		l_nostril = [lms[i] for i in [*range(12,18)]]
		base = l_nostril[1]
		top = l_nostril[-2]
		l_nostril, l_rot = rotate_vertically(label, img, l_nostril, base, top)
		r_nostril = []
		r_rot = 0

	elif abs(pose) == 60:
		l_nostril = [lms[i] for i in [*range(11,17)]]
		base = l_nostril[0]
		top = l_nostril[-3]
		l_nostril, l_rot = rotate_vertically(label, img, l_nostril, base, top)
		r_nostril = []
		r_rot = 0

	return [l_nostril, r_nostril], [l_rot, r_rot]

# ==============================================================================

def get_mouth(img, lms, pose):
	label = 'mouth_%d' % pose
	if abs(pose) == 0:
		return [], []
	elif abs(pose) == 30:
		mouth = [lms[i] for i in [*range(25,28)]]
		base = mouth[ -1]
		top = mouth[0]
		mouth, rot = rotate_horizontaly(label,img, mouth, base, top)

		if len(mouth) != 0:
			cv.imwrite(os.path.join(EX_FOLDER, '%s_%.2f.jpg' % ('mouth', rot)), mouth)


		return [mouth], [rot]
	elif abs(pose) == 60:
		mouth = [lms[i] for i in [*range(7,11)]]
		base = mouth[0]
		top = mouth[-1]
		mouth, rot = rotate_horizontaly(label, img,mouth, base, top)

		return [mouth], [rot]

# ==============================================================================

"""
def get_cheeks(img, lms, pose):

	if abs(pose) == 60: x
		cheek = [lms[i] for i in [*range(3,7)]]
		base = l_eye[0]
		top = l_eye[-3]
		rotate_vertically(img, points, base, top)
"""

# ==============================================================================

def get_all_features(list_imgs, O, PPC, CPB, mean_ratio = 0):
	if mean_ratio == 0:
		if len(list_imgs[0]) != 1:
			ratios_0 = [np.shape(img[0])[0]/np.shape(img[0])[1] for img in list_imgs if img[0] != []]
			ratios_1 = [np.shape(img[1])[0]/np.shape(img[1])[1] for img in list_imgs if img[1] != []]
			mean_ratio = [np.mean(ratios_0), np.mean(ratios_1)]
		else:
			ratios = [np.shape(img)[0]/np.shape(img)[1] for img in list_imgs]
			mean_ratio = np.mean(ratios)

	all_HOG = []
	for img in list_imgs:
		if len(list_imgs[0]) != 1:
			HOGs = []
			left = img[0]
			right = img[1]

			if left == []:
				left = np.zeros((200,200, 3))
			if right == []:
				right = np.zeros((200, 200, 3))

			for i, ratio in [[left, mean_ratio[0]], [right, mean_ratio[1]]]:
				 new_img = resize_img(i,ratio)

				 fd =  hog(new_img,  orientations= O, pixels_per_cell=(PPC,PPC), cells_per_block=(CPB, CPB), multichannel=True)
				 HOGs.append(fd)

			all_HOG.append(np.hstack(HOGs))
		else:
			if img == []:
				img = np.zeros((200,200, 3))
			print(np.shape(img))
			img = np.squeeze(img)
			new_img = resize_img(img,mean_ratio)
			fd =  hog(new_img,  orientations= O, pixels_per_cell=(PPC,PPC), cells_per_block=(CPB, CPB), multichannel=True)
			all_HOG.append(fd)



    #return the features, a 4096 vector
	#return all_HOG, mean_ratio, features_CNN
	return all_HOG, mean_ratio


