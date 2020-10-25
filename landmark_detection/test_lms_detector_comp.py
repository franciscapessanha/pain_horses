#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 10:12:25 2020

@author: franciscapessanha
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:05:55 2020
@author: franciscapessanha
"""

#%%============================================================================
#                       IMPORTS AND INITIALIZATIONS
#%%============================================================================

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
from train_lms_detector_comp import sorted_image_import, test_eval

DATASET = os.path.join(os.getcwd(),'..', 'dataset')
ABS_POSE_FITTER = os.path.join(DATASET,'abs_pose_complete')


ANIMAL = 'horse'
if ANIMAL == 'donkey':
    ABS_POSE = os.path.join(DATASET,'abs_pose_donkeys')
elif ANIMAL == 'horse':
    ABS_POSE = os.path.join(DATASET,'abs_pose_complete')

for folder in ['frontal', 'tilted', 'profile']:
    path = os.path.join(ABS_POSE, folder, 'gt')

    if os.path.exists(path) is not True:
        os.mkdir(path)

prefix = 'frontal_pert_30.pkl'

if ANIMAL == 'donkey':
    path_to_images = os.path.join(ABS_POSE,'frontal/')
elif ANIMAL == 'horse':
    path_to_images = os.path.join(ABS_POSE,'frontal/test/')

images_30, files_30 = sorted_image_import(path_to_images)

for folder in ['ert','sdm', 'mean']:
    if os.path.exists(os.path.join(ABS_POSE, 'frontal', folder)) is not True:
        os.mkdir(os.path.join(ABS_POSE, 'frontal', folder))

print('ERT - 0 to 30 degrees  \n=========================')
fitter_path = os.path.join(ABS_POSE_FITTER , 'frontal','train', 'fitters', 'ert_' + prefix)
fitter = mio.import_pickle(fitter_path)
errors = test_eval(fitter, images_30, files_30, pose = '0', save_images = True, folder = os.path.join(ABS_POSE, 'frontal','ert'), gt =os.path.join(ABS_POSE, 'frontal','gt'))

"""
print('SDM - 0 to 30 degrees \n=========================')
fitter_path = os.path.join(ABS_POSE_FITTER , 'frontal','train', 'fitters', 'sdm_' + prefix)
fitter = mio.import_pickle(fitter_path)
errors = test_eval(fitter, images_30, files_30, pose = '0',  save_images = True, folder = os.path.join(ABS_POSE, 'frontal','sdm'), gt =os.path.join(ABS_POSE, 'frontal','gt'))

print('Mean - 0 to 30 degrees  \n=========================')
fitter_path = os.path.join(ABS_POSE_FITTER , 'frontal','train','fitters', 'mean_' + prefix)
fitter = mio.import_pickle(fitter_path)
errors  = test_eval(fitter,images_30,  files_30, mean = True, pose = '0',  save_images = True, folder = os.path.join(ABS_POSE, 'frontal','mean'), gt =os.path.join(ABS_POSE, 'frontal','gt'))
"""


prefix = 'tilted_pert_30.pkl'

if ANIMAL == 'donkey':
    path_to_images = os.path.join(ABS_POSE,'tilted/')
elif ANIMAL == 'horse':
    path_to_images = os.path.join(ABS_POSE,'tilted/test/')

images_60, files_60 = sorted_image_import(path_to_images)

for folder in ['ert','sdm', 'mean']:
    if os.path.exists(os.path.join(ABS_POSE, 'tilted', folder)) is not True:
        os.mkdir(os.path.join(ABS_POSE, 'tilted', folder))


print('ERT - 30 to 60 degrees \n=========================')
fitter_path = os.path.join(ABS_POSE_FITTER , 'tilted','train', 'fitters', 'ert_' + prefix)
fitter = mio.import_pickle(fitter_path)
errors = test_eval(fitter, images_60, files_60, pose = '30',  save_images = True, folder = os.path.join(ABS_POSE, 'tilted','ert'), gt =os.path.join(ABS_POSE, '30_60','gt'))

"""
print('SDM - 30 to 60 degrees \n=========================')
fitter_path = os.path.join(ABS_POSE_FITTER , 'tilted','train', 'fitters', 'sdm_' + prefix)
fitter = mio.import_pickle(fitter_path)
errors = test_eval(fitter, images_60, files_60, pose = '30',  save_images = True,  folder = os.path.join(ABS_POSE, 'tilted','sdm'), gt =os.path.join(ABS_POSE, '30_60','gt'))

print('Mean - 30 to 60 degrees  \n=========================')''
fitter_path = os.path.join(ABS_POSE_FITTER , 'tilted','train','fitters', 'mean_' + prefix)
fitter = mio.import_pickle(fitter_path)
errors  = test_eval(fitter,images_60,  files_60, mean = True, pose = '30', save_images = True,  folder = os.path.join(ABS_POSE, 'tilted','mean'),gt =os.path.join(ABS_POSE, '30_60','gt'))
"""

ANIMAL = 'horse'
if ANIMAL == 'donkey':
    ABS_POSE = os.path.join(DATASET,'abs_pose_donkeys')
elif ANIMAL == 'horse':
    ABS_POSE = os.path.join(DATASET,'abs_pose_complete')

prefix = 'profile_pert_30.pkl'

if ANIMAL == 'donkey':
    path_to_images = os.path.join(ABS_POSE,'profile/')
elif ANIMAL == 'horse':
    path_to_images = os.path.join(ABS_POSE,'profile/test/')

images_90, files_90 = sorted_image_import(path_to_images)

for folder in ['ert','sdm', 'mean']:
    if os.path.exists(os.path.join(ABS_POSE, 'profile', folder)) is not True:
        os.mkdir(os.path.join(ABS_POSE, 'profile', folder))
print('ERT - 60 to 90 degrees \n=========================')
fitter_path = os.path.join(ABS_POSE_FITTER , 'profile','train', 'fitters', 'ert_' + prefix)
fitter = mio.import_pickle(fitter_path)
errors = test_eval(fitter, images_90, files_90, pose = '60',  save_images = True, folder = os.path.join(ABS_POSE, 'profile','ert'),gt =os.path.join(ABS_POSE, 'profile','gt'))
"""
print('SDM - 60 to 90 degrees \n=========================')
fitter_path = os.path.join(ABS_POSE_FITTER , 'profile','train', 'fitters', 'sdm_' + prefix)
fitter = mio.import_pickle(fitter_path)
errors = test_eval(fitter, images_90, files_90, pose = '60',  save_images = True, folder = os.path.join(ABS_POSE, 'profile','sdm'),gt =os.path.join(ABS_POSE, '60_90','gt'))

print('Mean - 60 to 90 degrees  \n=========================')
fitter_path = os.path.join(ABS_POSE_FITTER , 'profile','train', 'fitters', 'mean_' + prefix)
fitter = mio.import_pickle(fitter_path)
errors  = test_eval(fitter,images_90, files_90, mean = True, pose = '60',  save_images = True, folder = os.path.join(ABS_POSE, 'profile','mean'),gt =os.path.join(ABS_POSE, '60_90','gt'))
"""