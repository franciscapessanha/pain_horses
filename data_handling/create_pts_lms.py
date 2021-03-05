#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from sklearn.model_selection import train_test_split
import shutil

BIG_SIDE = 600

DATASET = os.path.join(os.getcwd(),'..','dataset')
ANGLES =  os.path.join(DATASET, '3D_annotations', 'angles')

LMS_SYSTEM = 'complete' # complete or absolute


if LMS_SYSTEM == 'complete':
    ABS_POSE = os.path.join(DATASET,'abs_pose_complete')
elif LMS_SYSTEM == 'absolute':
     ABS_POSE = os.path.join(DATASET,'abs_pose')

if not os.path.exists(ABS_POSE):
    os.mkdir(ABS_POSE)

for folder in ['frontal', 'tilted','profile']:
    path = os.path.join(ABS_POSE,folder)
    if not os.path.exists(path):
        os.mkdir(path)

    for sub_folder in ['train', 'test']:
        sub_path = os.path.join(path,sub_folder)
        if not os.path.exists(sub_path):
            os.mkdir(sub_path)

data = pickle.load(open(os.path.join(DATASET, 'lms_annotations.pkl'), "rb"))

# ==============================================================================

def crop_image(img, lms, pose):

    if pose < 0:
        img = cv.flip(img, 1)
        h, w = np.shape(img)[:2]
        mirror_lms = []
        for pt in lms:
            new_pt = [w - pt[0], pt[1]]
            mirror_lms.append(new_pt)

        lms = np.vstack(mirror_lms)

    error = 0.10
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

    lms_crop = []
    for pt in lms:
        new_pt = ((pt[0] - x_min), (pt[1] - y_min))
        lms_crop.append(new_pt)

    lms_crop = np.vstack(lms_crop)

    return img_crop, lms_crop

# ==============================================================================
def resize_img(img, lms, ratio):
    init_h, init_w = np.shape(img)[:2]
    if ratio < 1:
        width = BIG_SIDE
        height = BIG_SIDE * ratio
    else:
        height = BIG_SIDE
        width = BIG_SIDE / ratio

    img_resize = cv.resize(img, (int(width), int(height)))
    lms_resize = []
    for pt in lms:
        new_pt = (pt[0]  * width/init_w, pt[1] * height/init_h)
        #new_pt = ((pt[0] - x_min), (pt[1] - y_min))
        lms_resize.append(new_pt)

    lms_resize = np.vstack(lms_resize)
    return img_resize, lms_resize

# ==============================================================================
def create_pts(lms, pts_path):
    with open(pts_path, 'w') as pts_file:
        pts_file.write('version: 1\n')
        pts_file.write('n_points:  ' + str(len(lms)) + '\n')
        pts_file.write('{\n')
        for (x, y) in lms:
            pts_file.write(str(int(x)) + ' ' + str(int(y)) + '\n')
        pts_file.write('}')

# ==============================================================================
def save_abs_imgs(train_path, test_path, abs_folder, indexes):
    dimensions = []
    all_imgs = []
    all_lms = []
    all_names = []
    for path in train_path:
        img_name = path.split('/')[-1].split('.')[0]
        all_names.append(img_name)
        img_info = data.values[int(img_name) -1]
        pose = img_info[2]
        lms = img_info[4]

        img = cv.imread(path)
        img, lms = crop_image(img, lms, pose)
        all_imgs.append(img)
        all_lms.append(lms)
        img_shape = np.shape(img)[:2]

        dimensions.append([img_shape[0], img_shape[1], img_shape[0]/img_shape[1]])

    ratio = np.mean(dimensions, axis = 0)[-1]

    for lms, img, img_name in zip(all_lms, all_imgs, all_names):
        img_resize, lms_resize = resize_img(img, lms, ratio)
        cv.imwrite(os.path.join(abs_folder, 'train', img_name + '.png'), img_resize)
        abs_lms = np.vstack([lms_resize[i] for i in indexes])
        #abs_lms = lms_resize
        create_pts(abs_lms, os.path.join(abs_folder, 'train', img_name + '.pts'))

    for path in test_path:
        img_name = path.split('/')[-1].split('.')[0]
        img_info = data.values[int(img_name) -1]
        pose = img_info[2]
        lms = img_info[4]

        img = cv.imread(path)
        img, lms = crop_image(img, lms, pose)
        img_resize, lms_resize = resize_img(img, lms, ratio)
        cv.imwrite(os.path.join(abs_folder, 'test', img_name + '.png'), img_resize)
        if LMS_SYSTEM == 'complete':
            abs_lms = lms_resize
        elif LMS_SYSTEM == 'absolute':
            abs_lms = np.vstack([lms_resize[i] for i in indexes])
        create_pts(abs_lms, os.path.join(abs_folder, 'test', img_name + '.pts'))

def main():
    for label , indexes in [['frontal',[0, 2, 4, 5, 7, 9, *range(10,35), *range(43,46)]], ['tilted', [0, 2, 4, 5, 7, 9,  *range(10,33)]], ['profile', [0, 2, 5, *range(6,10),*range(12,38)]]]:
        train = glob.glob(os.path.join(DATASET, 'cross_val', label, 'train', '*.jpg'))
        test = glob.glob(os.path.join(DATASET, 'cross_val', label, 'test', '*.jpg'))
        abs_folder = os.path.join(ABS_POSE, label)
        save_abs_imgs(train, test, abs_folder, indexes)


main()
