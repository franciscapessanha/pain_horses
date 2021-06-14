

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import pickle
import cv2 as cv
import trimesh
import glob




EX_FOLDER = os.path.join(os.getcwd(), 'examples')
N_FOLDS = 3

ABS_POSE = os.path.join(os.getcwd(), '..', 'dataset','abs_pose')
if not os.path.exists(ABS_POSE):
    os.mkdir(ABS_POSE)


TEMP = os.path.join(os.getcwd(), 'temp')

if not os.path.exists(TEMP):
    os.mkdir(TEMP)
def read_pts(filename):
    return np.loadtxt(filename, comments=("version:", "n_points:", "{", "}"))

n = 0
for k in range(N_FOLDS):
    for pose  in ['frontal', 'tilted', 'profile']:
        images = glob.glob(os.path.join(ABS_POSE, pose, 'train', '*.png'))
        images_aug = glob.glob(os.path.join(ABS_POSE, pose, 'form2_data_0.3_%d' %k, '*.png'))
        for i in range(10):
            n += 1
            img_path = images[i]
            img_aug_path = images_aug[i]
            img = cv.imread(img_path)
            img_aug = cv.imread(img_aug_path)

            lms = read_pts(img_path.replace('png', 'pts'))
            lms_aug = read_pts(img_aug_path.replace('png', 'pts'))
            if i == 0:
                colors = [tuple(np.random.choice(range(256), size=3)) for i in range(len(lms))]

            for j,pt in enumerate(lms):
                cv.circle(img, (int(pt[0]), int(pt[1])), 5, (int(colors[j][0]),int(colors[j][1]), int(colors[j][2])), thickness=-1, lineType=cv.LINE_AA)

            for j,pt in enumerate(lms_aug):
                cv.circle(img_aug, (int(pt[0]), int(pt[1])), 5, (int(colors[j][0]),int(colors[j][1]), int(colors[j][2])), thickness=-1, lineType=cv.LINE_AA)

            cv.imwrite(os.path.join(TEMP, '%s_%d.png' % (pose,n)), img)
            cv.imwrite(os.path.join(TEMP, '%s_%d_aug.png' % (pose,n)), img_aug)
