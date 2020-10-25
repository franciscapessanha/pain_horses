#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 10:49:33 2020

@author: franciscapessanha
"""

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
from load_obj import *
from random import randint
from transformations import *
from pylab import *
import scipy.linalg as lin
import trimesh
import utils
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import random
from scipy.spatial.transform import Rotation as Rot

DATASET = os.path.join(os.getcwd(), '..', 'dataset')
MODELS =  os.path.join(DATASET, '3D_annotations', 'models')

HEAD_OBJ = os.path.join(MODELS,'head_TOSCA.obj')

EX_FOLDER = os.path.join(os.getcwd(), 'examples')

vertices, triangles = load_obj(HEAD_OBJ)

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

#%%%

mean_vertices = np.mean(vertices, axis = 0)
vertices = np.vstack([v - mean_vertices for v in vertices])
vertices = rotate_points(vertices)

#R_y = -yaw_list[i]
#R_x = random.sample(roll_list,1)[0]
#R_z = random.sample(pitch_list,1)[0]

R_y = 0
R_x = 0
R_z = 0



#vertices = rotate_points(vertices)

R_matrix = angles_to_rotmat(R_z, R_y, R_x)

vertices_rot = []
for v in vertices:
    vertices_rot.append(np.dot(R_matrix, v))

vertices_rot = np.vstack(vertices_rot)

RT = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=-1)
RT_4x4 = np.concatenate([RT, np.array([0., 0., 0., 1.])[None, :]], 0)
RT_4x4 = np.linalg.inv(RT_4x4)
RT_4x4 = RT_4x4 @ np.diag([1, -1, -1, 1])


R_matrix = angles_to_rotmat(R_z, R_y, R_x)
vertices_rot = []
for v in vertices:
    vertices_rot.append(np.dot(R_matrix, v))


faces = []
for tri in triangles:
	values = tri.split(' ')[:-1]
	f = np.hstack([i.split('//')[0] for i in values])[1:]
	f = f.astype(np.int)
	f = [i - 1 for i in f]
	faces.append(f)


"""
faces = []
for tri in triangles:
	values = tri.split(' ')
	f = [int(i) - 1 for i in values[1:]]
	faces.append(f)
"""
R_y = 0
R_z = 0
R_x = 0

R_matrix = angles_to_rotmat(R_z, R_y, R_x)

RT = np.concatenate((R_matrix, np.zeros((3, 1))), axis=-1)
RT_4x4 = np.concatenate([RT, np.array([0., 0., 0., 1.])[None, :]], 0)
RT_4x4 = np.linalg.inv(RT_4x4)
RT_4x4 = RT_4x4 @ np.diag([1, -1, -1, 1])

mesh = trimesh.Trimesh(vertices = vertices, faces=faces, process=True)
scene = mesh.scene()
scene.camera_transform = scene.camera.look_at(vertices, rotation = RT_4x4)

mesh_rot = trimesh.Trimesh(vertices = vertices_rot, faces=faces, process=True)
for f in range(len(mesh_rot.faces)):
	mesh_rot.visual.face_colors[f] = np.asarray([0,0,255, 200])#trimesh.visual.random_color()


R_y = 9.50
R_z = 11.43
R_x = 9.25

R_matrix = angles_to_rotmat(R_z, R_y, R_x)
vertices_rot = []
for v in vertices:
    vertices_rot.append(np.dot(R_matrix, v))

R_y = 0
R_z = 0
R_x = 0

mesh_rot_2 = trimesh.Trimesh(vertices = vertices_rot, faces=faces, process=True)
for f in range(len(mesh_rot_2.faces)):
	mesh_rot_2.visual.face_colors[f] = np.asarray([255,0,0, 200])#trimesh.visual.random_color()




scene_rot = mesh_rot.scene()
scene_rot.add_geometry(mesh_rot_2)
#scene_rot.add_geometry(trimesh.creation.axis(origin_size=1, transform=None, origin_color=(255,255,0), axis_radius=1, axis_length=30))
scene_rot.camera.resolution = np.asarray([800,800])
scene_rot.camera_transform = scene.camera.look_at(vertices_rot, rotation = RT_4x4)
name = 'yaw_%d_pitch_%d_roll_%d.png' % (R_y,R_x,R_z)

png = scene_rot.save_image(resolution = scene_rot.camera.resolution, visible = True)
with open(os.path.join(EX_FOLDER, name) , 'wb') as f:
	f.write(png)
	f.close()


"""
plt3d = plt.figure().gca(projection='3d')
plt3d.scatter(vertices[:,0], vertices[:,1], vertices[:,2], c='b', alpha = 0.05)
plt.show()

plt3d = plt.figure().gca(projection='3d')
plt3d.scatter(vertices_rot[:,0], vertices_rot[:,1], vertices_rot[:,2], c='r', alpha = 0.05)
plt.show()
"""
#plt3d.scatter(outline_lms[:,1], outline_lms[:,2], outline_lms[:,3], c='r', alpha = 1)
#plt3d.scatter(lms_model[:,1], lms_model[:,2], lms_model[:,3], c='g', alpha = 1)