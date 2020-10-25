#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 14:47:06 2020

@author: franciscapessanha
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ObjLoader import *
import numpy as np
import os
import pickle
import cv2 as cv
from load_obj import *
from random import randint
from tps import *
from pylab import *
import scipy.linalg as lin



HEAD_OBJ = 'cropped_head.obj'
LMS_FILE = 'ears_eyes_noistrils_mouth.txt'

DATASET = os.path.join(os.getcwd(), '..', 'dataset')
data = pickle.load(open(os.path.join(DATASET, 'lms_annotations.pkl'), "rb"))

ex_folder = os.path.join(os.getcwd(), 'examples')

if not os.path.exists(ex_folder):
	os.mkdir(ex_folder)


# DATASET  - TOSCA High Resolution: http://tosca.cs.technion.ac.il/book/resources_data.html

triangles = []
for line in open(os.path.join(os.getcwd(),'horse0.tri'), 'r'):
	line = line.split()
	line = [int(l) for l in line]
	triangles.append(line)

triangles = np.vstack(triangles)

vertices = []
for line in open(os.path.join(os.getcwd(),'horse0.vert'), 'r'):
	line = line.split()
	line = [float(l) for l in line]
	vertices.append(line)

vertices = np.vstack(vertices)

filepath_out = os.path.join(os.getcwd(),'horse_TOSCA.obj')
with open(filepath_out, 'w') as ofile:

	for v in vertices:
		# line = "v {v.x} {v.y} {v.z} " for full precision
		line = "v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n"
		#line = "vn 0.0 0.0 0.0\n"
		line = line.format(v=v)
		#ofile.write("vn 0.0 0.0 0.0 ")
		ofile.write(line)


	for f in triangles:
			line = "f {f[0]:d} {f[1]:d} {f[2]:d}\n"
			# obj vertex indices for faces start at 1 not 0 like blender.
			line = line.format(f=f)
			ofile.write(line)


# fig = plt.figure()
# ax = gca(projection='3d')
# ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2])
# show()