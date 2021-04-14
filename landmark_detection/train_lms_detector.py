#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 17:20:05 2020

@author: franciscapessanha
"""

#%%============================================================================
#             IMPORTS AND INITIALIZATIONS
#%============================================================================
import menpofit
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import menpo.io as mio
from functools import partial
from menpo.base import LazyList
from menpofit.dlib.fitter import DlibERT
from menpofit.sdm import RegularizedSDM
from menpo.feature import vector_128_dsift
from menpofit.error.base import euclidean_bb_normalised_error
import cv2 as cv
from menpo.shape import mean_pointcloud
import math
import pickle
from create_pts_lms import create_pts_files


DATASET = os.path.join(os.getcwd(), '..', 'dataset')
ANGLES =  os.path.join(DATASET, '3D_annotations', 'angles')


N_FOLDS = 3
AUG = 0.7

LMS_SYSTEM = 'absolute' # complete or absolute
if LMS_SYSTEM == 'absolute':
    ABS_POSE = os.path.join(DATASET,'abs_pose')
elif LMS_SYSTEM == 'complete':
    ABS_POSE = os.path.join(DATASET,'abs_pose_complete')

#%%============================================================================
#                             AUXILIAR FUNCTIONS
#==============================================================================

def sort_files(l):
  # sort files based on the image name (numerical order)
  def f(path):
    path = path.split('/')[-1]
    path = path.split('.')[0]
    return int(path)

  return sorted(l, key=f)

#==============================================================================

def sorted_image_import(folder):
  file_list = sort_files(glob.glob(folder + '/*.png'))
  """
  The default behaviour of import_images() imports and attaches the landmark
  files that share the same filename stem as the images. The group name of the
  attached landmarks is based on the extension of the landmark file. This means
  that it is much more convenient to locate the landmark files in the same directory
  as the image files - LANDMARK FILE, SAME FOLDER *.PTS
  """
  # Single image (and associated landmarks) importer
  # We will create a LazyList of partial objects; these objects have 3 read-only
  # arguments: func (mio.import_image) ; args (f) ; keywords;

  l = LazyList([partial(mio.import_image,f) for f in file_list])
  print(len(l), 'images imported from', folder)
  return l, file_list

#==============================================================================
#%%============================================================================
#                                 MODELS
#==============================================================================
def ERT(data, path_to_images, n_pert = 30, prefix = '', verbose = True):
  train = data[0][:]
  name_fitter = prefix + '.pkl'
  print('fitters/' + name_fitter)

  if os.path.exists(os.path.join(path_to_images, 'fitters')) is not True:
        os.mkdir(os.path.join(path_to_images, 'fitters'))

  fitter_path = os.path.join(path_to_images, 'fitters', name_fitter)
  if os.path.exists(fitter_path):
    if verbose:
      print('Loaded fitter', name_fitter)
    fitter = mio.import_pickle(fitter_path)
  else:
    if verbose:
      print('Training fitter', name_fitter)
    fitter = DlibERT(train, scales=(1), verbose=verbose, n_perturbations=n_pert)
    if verbose:
      print('Saving fitter', name_fitter)

    mio.export_pickle(fitter, fitter_path)

#==============================================================================
def new_SDM(data, path_to_images, n_pert = 30, prefix = '', verbose = True):
    train = data[0][:]

    name_fitter = prefix + '.pkl'
    print('fitters/' + name_fitter)

    if os.path.exists(os.path.join(path_to_images, 'fitters')) is not True:
      os.mkdir(os.path.join(path_to_images, 'fitters'))

    fitter_path = os.path.join(path_to_images, 'fitters', name_fitter)
    if os.path.exists(fitter_path):
      if verbose:
        print('Loaded fitter', name_fitter)
      fitter = mio.import_pickle(fitter_path)
    else:
      if verbose:
        print('Training fitter', name_fitter)
      fitter = RegularizedSDM(
              train,
              verbose=True,
              group='PTS',
              diagonal=200,
              n_perturbations=n_pert,
              n_iterations=2,
              patch_features=vector_128_dsift,
              patch_shape=(24, 24),
              alpha=10
            )

      if verbose:
        print('Saving fitter', name_fitter)

      mio.export_pickle(fitter, fitter_path)


def fit_mean_shape(data, prefix, path_to_images, verbose = True):
    if os.path.exists(os.path.join(path_to_images, 'fitters')) is not True:
        os.mkdir(os.path.join(path_to_images, 'fitters'))
    train = data[0][:]
    name_fitter = prefix + '.pkl'
    print('fitters/' + name_fitter)

    fitter_path = os.path.join(path_to_images, 'fitters', name_fitter)
    print(fitter_path)
    if os.path.exists(fitter_path):
      if verbose:
        print('Loaded fitter', name_fitter)
      mean_shape = mio.import_pickle(fitter_path)
    else:
      print('Train shape')
      mean_shape = mean_pointcloud([image.landmarks['PTS']for image in train])
      mio.export_pickle(mean_shape, fitter_path)


def train_model(path_to_images, prefix, n_pert):
    if MODE == 'cross_val':
        images, files = sorted_image_import(path_to_images)
        for k in range(N_FOLDS):
            fold = np.vstack(pickle.load(open(os.path.join(ANGLES, '%s_pain_val_fold_%d_angles.pickle' % (prefix, k)), 'rb')))
            indexes_train = [i for i in range(len(files)) if files[i].split('/')[-1].split('.')[0] + '.jpg' not in fold[:,0]]
            file_list = [files[i] for i in indexes_train]
            images = LazyList([partial(mio.import_image,f) for f in file_list])
            landmarks = images.map(lambda x: x.landmarks) #Extracts the landmarks (associated with each image)
            images_aug, files_aug = sorted_image_import(os.path.join('/', *path_to_images.split('/')[:-1], 'data_%.1f_%d' % (AUG, k)))
            landmarks_aug = images_aug.map(lambda x: x.landmarks) #Extracts the landmarks (associated with each image)
            umages = images + images_aug
            landmarks = landmarks + landmarks_aug
            ERT((images, landmarks), path_to_images, n_pert= n_pert, prefix = ('ert_fold_' + str(k) +  '_pert_%d' %n_pert + '_aug_' + str(AUG)), verbose = True)
            fit_mean_shape((images, landmarks), ('mean_' + prefix + '_pert_%d' %n_pert), path_to_images, verbose = True)
            new_SDM((images, landmarks), path_to_images, n_pert= n_pert, prefix = ('sdm_' + prefix + '_pert_%d' %n_pert), verbose = True)

    elif MODE == 'final_model':
        images, files = sorted_image_import(path_to_images)
        landmarks = images.map(lambda x: x.landmarks) #Extracts the landmarks (associated with each image)
        print('len landmarks: ', len(landmarks))
        images_aug, files_aug = sorted_image_import(os.path.join('/', *path_to_images.split('/')[:-1], 'data_%.1f_%d' (AUG, k)))
        landmarks_aug = images_aug.map(lambda x: x.landmarks) #Extracts the landmarks (associated with each image)
        images = images + images_aug
        landmarks = landmarks + landmarks_aug
        ERT((images, landmarks), path_to_images, n_pert= n_pert, prefix = ('ert_' + prefix + '_pert_%d' %n_pert + 'aug_' + str(AUG)), verbose = True)
        new_SDM((images, landmarks), path_to_images, n_pert= n_pert, prefix = ('sdm_' + prefix + '_pert_%d' %n_pert), verbose = True)
        fit_mean_shape((images, landmarks), ('mean_' + prefix + '_pert_%d' %n_pert), path_to_images, verbose = True)



#%%============================================================================
#                       RUN
#=============================================================================

data_aug = False
LMS_SYSTEM = 'absolute'
MODE = 'cross_ val'
if LMS_SYSTEM == 'absolute':
    n_pert = 80
if LMS_SYSTEM == 'complete':
    n_pert = 100

def main():
    create_pts_files(LMS_SYSTEM)
    if MODE == 'cross_val':
        for n_pert in [60, 70, 80, 90, 100]:
            train_model(os.path.join(ABS_POSE,'frontal', 'train'), 'frontal', n_pert)
            train_model(os.path.join(ABS_POSE,'tilted',  'train'), 'tilted', n_pert)
            train_model(os.path.join(ABS_POSE,'profile',  'train'), 'profile', n_pert)

    elif MODE == 'final_model':
        train_model(os.path.join(ABS_POSE,'frontal', 'train'), 'frontal', n_pert)
        train_model(os.path.join(ABS_POSE,'tilted', 'train'), 'tilted', n_pert)
        train_model(os.path.join(ABS_POSE,'profile', 'train'), 'profile', n_pert)
main()
