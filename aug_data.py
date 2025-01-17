#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 09:48:44 2020

@author: franciscapessanha
"""


# %%============================================================================
#                          IMPORTS AND INITIALIZATIONS
# ==============================================================================

import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import pickle
import cv2 as cv
import trimesh
import glob
from landmark_detection.create_pts_lms import crop_image, resize_img, create_pts
from data_augmentation.load_obj import *
from data_augmentation.transformations import *
import math
#from data_augmentation.utils import *
import random
from scipy.spatial import distance

DATASET = os.path.join(os.getcwd(), 'dataset')
COLORS =  os.path.join(DATASET, '3D_annotations', 'colors')
SHAPES =  os.path.join(DATASET, '3D_annotations', 'shapes')
ANGLES =  os.path.join(DATASET, '3D_annotations', 'angles')
MODELS =  os.path.join(DATASET, '3D_annotations', 'models')


EX_FOLDER = os.path.join(os.getcwd(), 'examples_data_aug')
if not os.path.exists(EX_FOLDER):
    os.mkdir(EX_FOLDER)

ABS_POSE = os.path.join(os.getcwd(), 'dataset','abs_pose')
N_FOLDS = 3


if not os.path.exists(ABS_POSE):
    os.mkdir(ABS_POSE)

for folder in ['frontal', 'tilted','profile']:
    path = os.path.join(ABS_POSE,folder)
    if not os.path.exists(path):
        os.mkdir(path)

    for k in range(N_FOLDS):
        for AUG in  [0.07, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.1, 1.2] :
            sub_path = os.path.join(path,'error3_data_%.2f_%d' % (AUG, k))
            if not os.path.exists(sub_path):
                os.mkdir(sub_path)

    for AUG in  [0.07, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.1, 1.2] :
        sub_path = os.path.join(path,'error3_data_%.2f_final' % (AUG))
        if not os.path.exists(sub_path):
            os.mkdir(sub_path)

TEMP = os.path.join(os.getcwd(), 'temp')

if not os.path.exists(TEMP):
    os.mkdir(TEMP)

def rotate_points(points):
    R_z = 0
    R_y = 0
    R_x = -90

    R_matrix = angles_to_rotmat(R_z, R_y, R_x)
    points_rot = []
    for pt in points:
        points_rot.append(np.dot(R_matrix, pt))
    points_rot = np.vstack(points_rot)
    return points_rot

def find_closest_point(silhouete, point):
    closest_index = distance.cdist(np.vstack(silhouete), np.vstack([point])).argmin()

    return silhouete[closest_index]

formula = 'median'
MODE = 'final_model'
#%%
BACKGROUND = glob.glob(os.path.join(DATASET, 'flickr_backgrounds', '*.png'))


angles = glob.glob(os.path.join(ANGLES, '*.pickle'))
shapes = glob.glob(os.path.join(SHAPES, '*.obj'))
colors_list = glob.glob(os.path.join(COLORS, '*.pickle'))

# ['tilted', 600/333],
for AUG in [0.1]:
    #for pose,ratio in [['frontal', 600/270]]:
    #'tilted', 600/333]
    #['frontal', 600/270]
    #for pose,ratio in [['frontal', 600/270]]:
    #for pose, ratio in [['profile', 600/381]]:
    #for pose, ratio in [['profile', 600/381]]:
    for pose, ratio in [['tilted', 600/333]]:
        print('AUG = %.2f'% AUG)
        print('label: ', pose)
        #yaw_list = np.load(open(os.path.join(os.getcwd(), 'landmark_detection', '%s_%s_%s_fold_%d.pickle' % (pose, AUG, 'yaw', k)), 'rb'), allow_pickle = True)
        #pitch_list = np.load(open(os.path.join(os.getcwd(), 'landmark_detection', '%s_%s_%s_fold_%d.pickle' % (pose, AUG, 'pitch', k)), 'rb'), allow_pickle = True)
        #roll_list = np.load(open(os.path.join(os.getcwd(), 'landmark_detection', '%s_%s_%s_fold_%d.pickle' % (pose, AUG, 'roll', k)), 'rb'), allow_pickle = True)
        if MODE == 'cross_val':
            for k in range(N_FOLDS):
                print('k: ', k)

                if formula == 'median':
                    yaw_list = np.load(open(os.path.join(os.getcwd(), 'landmark_detection', 'data_aug', 'error3_%s_aug_%.1f_%s_fold_%d.pickle' % (pose, AUG, 'yaw', k)), 'rb'), allow_pickle = True)
                    pitch_list = np.load(open(os.path.join(os.getcwd(), 'landmark_detection','data_aug', 'error3_%s_aug_%.1f_%s_fold_%d.pickle' % (pose, AUG, 'pitch', k)), 'rb'), allow_pickle = True)
                    roll_list = np.load(open(os.path.join(os.getcwd(), 'landmark_detection', 'data_aug', 'error3_%s_aug_%.1f_%s_fold_%d.pickle' % (pose, AUG, 'roll', k)), 'rb'), allow_pickle = True)
                    print('len yaw: ', len(yaw_list))
                    #if len(glob.glob(os.path.join(ABS_POSE, pose, 'data_%.1f_%d' % (AUG, k),  '*.png'))) == len(yaw_list):
                    #   break
                    #else:
                    random.shuffle(yaw_list)
                    random.shuffle(pitch_list)
                    random.shuffle(roll_list)

                if formula == 'mean':
                    yaw_list = np.load(open(os.path.join(os.getcwd(), 'landmark_detection', 'data_aug', 'error_%s_aug_%s_%s_fold_%d.pickle' % (pose, AUG, 'yaw', k)), 'rb'), allow_pickle = True)
                    pitch_list = np.load(open(os.path.join(os.getcwd(), 'landmark_detection','data_aug', 'error_%s_aug_%s_%s_fold_%d.pickle' % (pose, AUG, 'pitch', k)), 'rb'), allow_pickle = True)
                    roll_list = np.load(open(os.path.join(os.getcwd(), 'landmark_detection', 'data_aug', 'error_%s_aug_%s_%s_fold_%d.pickle' % (pose, AUG, 'roll', k)), 'rb'), allow_pickle = True)

                #if len(glob.glob(os.path.join(ABS_POSE, pose, 'mean_data_%.1f_%d' % (AUG, k),  '*.png'))) == len(yaw_list):
                #   break
                #else:
                random.shuffle(yaw_list)
                random.shuffle(pitch_list)
                random.shuffle(roll_list)


                print('len yaw list: ', len(yaw_list))
                if formula == 'mean':
                    folder = os.path.join(ABS_POSE, pose, 'error3_data_%.1f_%d' % (AUG, k))
                elif formula == 'median':
                    folder = os.path.join(ABS_POSE, pose, 'error3_data_%.1f_%d' % (AUG, k))
                for i in range(len(glob.glob(os.path.join(folder,  '*.png'))),len(yaw_list)):
                    print('%d / %d' % (i+1, len(yaw_list)))
                    colors =  pickle.load(open(random.choice(colors_list), 'rb'))
                    shape = random.choice(shapes)
                    vertices, triangles = load_obj(shape)

                    faces = []
                    for tri in triangles:
                        values = tri.split(' ')[:-1]
                        f = np.hstack([i.split('//')[0] for i in values])[1:]
                        f = f.astype(np.int)
                        f = [i - 1 for i in f]
                        faces.append(f)

                    """
                    vert = np.vstack([vertices[i] for i in pickle.load(open(os.path.join(MODELS, pose + '_indexes.pickle'), 'rb'))])
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c = 'r', alpha = 0.05)
                    ax.scatter(vert[:, 0], vert[:, 1], vert[:, 2], c = 'b')
                    plt.show()
                    """
                    path_bg = random.choice(list(BACKGROUND))
                    img_background = cv.imread(path_bg)

                    # [euler_angles_degrees[2], euler_angles_degrees[0], euler_angles_degrees[1]]  # roll, pitch, yaw
                    R_y = yaw_list[i]
                    R_x = roll_list[i]
                    R_z = pitch_list[i]

                    vertices = rotate_points(vertices)

                    R_matrix = angles_to_rotmat(R_z, R_y, R_x)

                    vertices_rot = []
                    for v in vertices:
                        vertices_rot.append(np.dot(R_matrix, v))

                    vertices_rot = np.vstack(vertices_rot)

                    RT = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=-1)
                    RT_4x4 = np.concatenate([RT, np.array([0., 0., 0., 1.])[None, :]], 0)
                    RT_4x4 = np.linalg.inv(RT_4x4)
                    RT_4x4 = RT_4x4 @ np.diag([1, -1, -1, 1])

                    mesh = trimesh.Trimesh(vertices = vertices_rot, faces=faces, process=True)

                    for f in range(len(mesh.faces)):
                         mesh.visual.face_colors[f] = np.asarray(colors[f])#trimesh.visual.random_color()


                    scene = mesh.scene()
                    scene.camera.K = np.load(shape.replace('obj', 'pickle'), 'rb', allow_pickle = True)
                    scene.camera.resolution = tuple([1600,  800])
                    scene.camera_transform = scene.camera.look_at(vertices_rot)

                    name = 'yaw_%d_pitch_%d_roll_%d.png' % (R_y,R_z, R_x)

                    png = scene.save_image(resolution = scene.camera.resolution, background = [0,0,0], visible = True)
                    with open(os.path.join(TEMP, name) , 'wb') as f:
                        f.write(png)
                        f.close()

                    img_load = cv.imread(os.path.join(TEMP, name))

                    imgray = cv.cvtColor(img_load, cv.COLOR_BGR2GRAY)
                    ret, thresh = cv.threshold(imgray, 0, 10, 0)
                    _, contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                    silhouete = sorted(contours, key=len)[-1]

                    #cv.drawContours(img_load, silhouete, -1, (0,255,0), 3)
                    #cv.imwrite('silhouete.jpg', img_load)

                    lms, _ = project_pts(vertices_rot, scene.camera.K, external_parameters = np.dot(np.linalg.inv(RT_4x4),np.linalg.inv(scene.camera_transform))[:3,:])

                    silhouete = np.vstack(silhouete)

                    if pose == 'frontal':
                        int_lms = np.vstack([lms[j] for j in pickle.load(open(os.path.join(MODELS, pose + '_indexes.pickle'), 'rb'))])
                        update_lms = int_lms.copy()
                        indexes = [0, 2, 3, 5, 30, 31, 32, 33]
                        ratio = 600/270
                        for ind in indexes:
                            update_lms[ind] = find_closest_point(silhouete, int_lms[ind])
                    elif pose == 'tilted':
                        int_lms = np.vstack([lms[j] for j in pickle.load(open(os.path.join(MODELS, pose + '_sim_indexes.pickle'), 'rb'))])
                        update_lms = int_lms.copy()
                        ratio = 600/333
                        indexes = [*range(0,6), 18, 19, *range(22, 26)]
                        for ind in indexes:
                            update_lms[ind] = find_closest_point(silhouete, [int_lms[ind]])
                    elif pose == 'profile':
                        int_lms = np.vstack([lms[j] for j in pickle.load(open(os.path.join(MODELS, pose + '_sim_indexes.pickle'), 'rb'))])
                        update_lms = int_lms.copy()
                        ratio = 600/381
                        indexes = [0, 1, 2, *range(26, 33)]
                        for ind in indexes:
                            update_lms[ind] = find_closest_point(silhouete, [int_lms[ind]])


                    int_lms = update_lms
                    img_crop, lms_crop = crop_image(img_load, int_lms, R_y)
                    img_resize, lms_resize = resize_img(img_crop, lms_crop, ratio)

                    background_h, background_w = np.shape(img_background)[:2]
                    head_h, head_w = np.shape(img_resize)[:2]

                    ratio_head = head_h/head_w
                    ratio_background = background_h/background_w

                    if background_h > background_w:
                        new_background_h = head_h
                        new_background_w = max(background_w * head_h / background_h, head_w)

                    if  background_h < background_w:
                        new_background_w = head_w
                        new_background_h = max(background_h * head_w / background_w, head_h)

                    resize_background = cv.resize(img_background, (int(new_background_w), int(new_background_h)))
                    resize_background = cv.GaussianBlur(resize_background,(5,5),0)

                    new_img = img_resize.copy()
                    for i_h in range(head_h - 1):
                        for i_w in range(head_w - 1):
                            px = img_resize[i_h, i_w, :]
                            if (px == np.asarray([0,0,0])).all(): # if pixel black
                                #print('enter')
                                new_img[i_h, i_w, :] = resize_background[i_h, i_w, :]

                    img_with_pts   = new_img.copy()

                    #for pt in lms_resize:
                    #    cv.circle(img_with_pts,tuple((int(pt[0]), int(pt[1]))), 3, (255,0,0), -1)

                    cv.imwrite(os.path.join(EX_FOLDER, pose +'_' + str(k) + '_' + str(i) + '.png'), img_with_pts)

                    cv.imwrite(os.path.join(folder, str(i) + '.png'), new_img)
                    create_pts(lms_resize, os.path.join(folder, str(i) + '.pts'))


        else:

            yaw_list = np.load(open(os.path.join(os.getcwd(), 'landmark_detection', 'data_aug', 'error3_%s_aug_%.2f_%s_final.pickle' % (pose, AUG, 'yaw')), 'rb'), allow_pickle = True)
            pitch_list = np.load(open(os.path.join(os.getcwd(), 'landmark_detection','data_aug', 'error3_%s_aug_%.2f_%s_final.pickle' % (pose, AUG, 'pitch')), 'rb'), allow_pickle = True)
            roll_list = np.load(open(os.path.join(os.getcwd(), 'landmark_detection', 'data_aug', 'error3_%s_aug_%.2f_%s_final.pickle' % (pose, AUG, 'roll')), 'rb'), allow_pickle = True)

            random.shuffle(yaw_list)
            random.shuffle(pitch_list)
            random.shuffle(roll_list)



            print('len yaw list: ', len(yaw_list))
            if formula == 'mean':
                folder = os.path.join(ABS_POSE, pose, 'error3_data_%.2f_final' % (AUG))
            elif formula == 'median':
                folder = os.path.join(ABS_POSE, pose, 'error3_data_%.2f_final' % (AUG))
            for i in range(len(glob.glob(os.path.join(folder,  '*.png'))),len(yaw_list)):
                print('%d / %d' % (i+1, len(yaw_list)))
                colors =  pickle.load(open(random.choice(colors_list), 'rb'))
                shape = random.choice(shapes)
                vertices, triangles = load_obj(shape)

                faces = []
                for tri in triangles:
                    values = tri.split(' ')[:-1]
                    f = np.hstack([i.split('//')[0] for i in values])[1:]
                    f = f.astype(np.int)
                    f = [i - 1 for i in f]
                    faces.append(f)

                """
                vert = np.vstack([vertices[i] for i in pickle.load(open(os.path.join(MODELS, pose + '_indexes.pickle'), 'rb'))])
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c = 'r', alpha = 0.05)
                ax.scatter(vert[:, 0], vert[:, 1], vert[:, 2], c = 'b')
                plt.show()
                """
                path_bg = random.choice(list(BACKGROUND))
                img_background = cv.imread(path_bg)

                # [euler_angles_degrees[2], euler_angles_degrees[0], euler_angles_degrees[1]]  # roll, pitch, yaw
                R_y = yaw_list[i]
                R_x = roll_list[i]
                R_z = pitch_list[i]

                vertices = rotate_points(vertices)

                R_matrix = angles_to_rotmat(R_z, R_y, R_x)

                vertices_rot = []
                for v in vertices:
                    vertices_rot.append(np.dot(R_matrix, v))

                vertices_rot = np.vstack(vertices_rot)

                RT = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=-1)
                RT_4x4 = np.concatenate([RT, np.array([0., 0., 0., 1.])[None, :]], 0)
                RT_4x4 = np.linalg.inv(RT_4x4)
                RT_4x4 = RT_4x4 @ np.diag([1, -1, -1, 1])

                mesh = trimesh.Trimesh(vertices = vertices_rot, faces=faces, process=True)

                for f in range(len(mesh.faces)):
                     mesh.visual.face_colors[f] = np.asarray(colors[f])#trimesh.visual.random_color()


                scene = mesh.scene()
                scene.camera.K = np.load(shape.replace('obj', 'pickle'), 'rb', allow_pickle = True)
                scene.camera.resolution = tuple([1600,  800])
                scene.camera_transform = scene.camera.look_at(vertices_rot)

                name = 'yaw_%d_pitch_%d_roll_%d.png' % (R_y,R_z, R_x)

                png = scene.save_image(resolution = scene.camera.resolution, background = [0,0,0], visible = True)
                with open(os.path.join(TEMP, name) , 'wb') as f:
                    f.write(png)
                    f.close()

                img_load = cv.imread(os.path.join(TEMP, name))

                imgray = cv.cvtColor(img_load, cv.COLOR_BGR2GRAY)
                ret, thresh = cv.threshold(imgray, 0, 10, 0)
                _, contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                silhouete = sorted(contours, key=len)[-1]

                #cv.drawContours(img_load, silhouete, -1, (0,255,0), 3)
                #cv.imwrite('silhouete.jpg', img_load)

                lms, _ = project_pts(vertices_rot, scene.camera.K, external_parameters = np.dot(np.linalg.inv(RT_4x4),np.linalg.inv(scene.camera_transform))[:3,:])

                silhouete = np.vstack(silhouete)

                if pose == 'frontal':
                    int_lms = np.vstack([lms[j] for j in pickle.load(open(os.path.join(MODELS, pose + '_indexes.pickle'), 'rb'))])
                    update_lms = int_lms.copy()
                    indexes = [0, 2, 3, 5, 30, 31, 32, 33]
                    ratio = 600/270
                    for ind in indexes:
                        update_lms[ind] = find_closest_point(silhouete, int_lms[ind])
                elif pose == 'tilted':
                    int_lms = np.vstack([lms[j] for j in pickle.load(open(os.path.join(MODELS, pose + '_sim_indexes.pickle'), 'rb'))])
                    update_lms = int_lms.copy()
                    ratio = 600/333
                    indexes = [*range(0,6), 18, 19, *range(22, 26)]
                    for ind in indexes:
                        update_lms[ind] = find_closest_point(silhouete, [int_lms[ind]])
                elif pose == 'profile':
                    int_lms = np.vstack([lms[j] for j in pickle.load(open(os.path.join(MODELS, pose + '_sim_indexes.pickle'), 'rb'))])
                    update_lms = int_lms.copy()
                    ratio = 600/381
                    indexes = [0, 1, 2, *range(26, 33)]
                    for ind in indexes:
                        update_lms[ind] = find_closest_point(silhouete, [int_lms[ind]])


                int_lms = update_lms
                img_crop, lms_crop = crop_image(img_load, int_lms, R_y)
                img_resize, lms_resize = resize_img(img_crop, lms_crop, ratio)

                background_h, background_w = np.shape(img_background)[:2]
                head_h, head_w = np.shape(img_resize)[:2]

                ratio_head = head_h/head_w
                ratio_background = background_h/background_w

                if background_h > background_w:
                    new_background_h = head_h
                    new_background_w = max(background_w * head_h / background_h, head_w)

                if  background_h < background_w:
                    new_background_w = head_w
                    new_background_h = max(background_h * head_w / background_w, head_h)

                resize_background = cv.resize(img_background, (int(new_background_w), int(new_background_h)))
                resize_background = cv.GaussianBlur(resize_background,(5,5),0)

                new_img = img_resize.copy()
                for i_h in range(head_h - 1):
                    for i_w in range(head_w - 1):
                        px = img_resize[i_h, i_w, :]
                        if (px == np.asarray([0,0,0])).all(): # if pixel black
                            #print('enter')
                            new_img[i_h, i_w, :] = resize_background[i_h, i_w, :]

                img_with_pts   = new_img.copy()

                #for pt in lms_resize:
                #    cv.circle(img_with_pts,tuple((int(pt[0]), int(pt[1]))), 3, (255,0,0), -1)

                cv.imwrite(os.path.join(EX_FOLDER, pose +'_' + str(k) + '_' + str(i) + '.png'), img_with_pts)

                cv.imwrite(os.path.join(folder, str(i) + '.png'), new_img)
                create_pts(lms_resize, os.path.join(folder, str(i) + '.pts'))


