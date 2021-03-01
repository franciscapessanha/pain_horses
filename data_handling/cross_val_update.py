#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 16:09:18 2021

@author: Pessa001
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:17:14 2020

@author: franciscapessanha
"""

# %%============================================================================
#      IMPORTS AND INITIALIZATIONS
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
from itertools import product
from collections import Counter
import math
from collections import Counter
import copy


PLOTS = os.path.join(os.getcwd(), 'plots')
if not os.path.exists(PLOTS):
    os.mkdir(PLOTS)

N_FOLDS = 3
DATASET = os.path.join(os.getcwd(), '..', 'dataset')
ANGLES =  os.path.join(DATASET, '3D_annotations', 'angles')

CROSS_VAL = os.path.join(os.getcwd(), '..', 'dataset','cross_val')

if not os.path.exists(CROSS_VAL):
    os.mkdir(CROSS_VAL)

for folder in ['frontal', 'tilted','profile']:
    path = os.path.join(CROSS_VAL,folder)
    if not os.path.exists(path):
        os.mkdir(path)

    for sub_folder in ['train', 'test']:
        sub_path = os.path.join(path,sub_folder)
        if not os.path.exists(sub_path):
            os.mkdir(sub_path)

data = pickle.load(open(os.path.join(DATASET, 'lms_annotations.pkl'), "rb"))

pain = pd.read_excel(os.path.join(DATASET, 'pain_annotations.xlsx'), index_col=0, engine='openpyxl')
horse_id = pain.values[:, 8] #just horses (that is why the 1855)

# Define cross validation based on horses ids
# ===================================================
horses_with_id = [[i+1, horse_id[i]] for i in range(len(horse_id[:1855])) if not math.isnan(horse_id[i])]
horses_with_id = np.vstack(horses_with_id)

unique_ids = np.unique(horses_with_id[:,1], axis=0)

all_groups =[]
for id_ in unique_ids:
    group = []
    for horse in horses_with_id:
        if horse[1] == id_:
            group.append(horse[0])
    all_groups.append(group)
# divide animals per id

all_groups.sort(key = len, reverse = True)

val_size = int((len(horses_with_id) * 0.70)/3)
test_size = int(len(horses_with_id) * 0.30)

desired_size = [val_size, val_size, val_size, test_size]

cross_val = [all_groups[i] for i in range(N_FOLDS + 1)] # this applies to all classes of horses
for i in range(N_FOLDS + 1): all_groups.pop(i)


for i, set_ in enumerate(cross_val):
    final_size = desired_size[i]
    current_size = len(set_)
    while current_size < final_size and len(all_groups) != 0:
        set_ = [set_, all_groups[-1]]
        set_ = np.hstack(set_)
        cross_val[i] = set_
        current_size = len(set_)
        all_groups.pop(-1)

#===========================================================================
#                                   AUXILIAR FUNCTIONS
# ==============================================================================

def closer_bin(pt, array):
    k = []
    for i in array:
        k.append(abs(pt - i))
    #print('k index: ', k.index(min(k)))
    return k.index(min(k))

def get_y(yaws, bins, counts):
    y = []
    for yaw in yaws:
        for i in range(len(bins) - 1):
            if yaw >= bins[i] and yaw < bins[i + 1]:
                y.append(i)
                break

    single_classes = np.where(counts == 1)[0]
    null_classes = np.where(counts == 0)[0]

    non_single_classes = [i for i in range(len(counts)) if i not in [*null_classes,*single_classes]]
    updated_y = []
    for label in y:
        if label in single_classes:
            updated_y.append(non_single_classes[closer_bin(label, non_single_classes)])
            print('replaced class: ', non_single_classes[closer_bin(label, non_single_classes)])
        else:
            updated_y.append(label)

    return updated_y


def get_train_test(angles, label):
    x = angles[:, 0]
    total_images = len(x)
    photonumbers = [int(i.split('.')[0]) for i in x]
    x_pain = [pain.values[i][1:7] for i in photonumbers]


    combinations = np.asarray(list(product([-1, 0, 1, 2], repeat = 6))) # all possible combinations of labels
    y_pain = np.vstack([np.where((combinations == pain).all(axis = 1))[0] for pain in x_pain])
    labels = np.unique(y_pain)
    #print('lenght labels: ', len(labels))
    y_pain = np.vstack([np.where((labels == pain[0]))[0] for pain in y_pain])
    #angles = np.vstack([[fix_angles(float(r),float(p),float(y))] for r,p, y in angles[:,1:]])
    yaws = angles[:, -1].astype(np.float)
    rolls = angles[:, -3].astype(np.float)
    pitchs = angles[:, -2].astype(np.float)

    plt.figure()
    yaw_counts, yaw_bins, _ = plt.hist(yaws, bins=18, range = [-90, 90])
    title = 'Yaw distribution - %s pose' % label
    plt.title(title)
    plt.xlabel('angle (degrees)')
    plt.ylabel('number of occurrences')
    plt.savefig(os.path.join(PLOTS, title))

    plt.figure()
    rolls_counts, rolls_bins, _ = plt.hist(rolls, bins=18, range = [-90, 90])
    title = 'Roll distribution - %s pose' % label
    plt.title(title)
    plt.xlabel('angle (degrees)')
    plt.ylabel('number of occurrences')
    plt.savefig(os.path.join(PLOTS, title))

    plt.figure()
    pitchs_counts, pitchs_bins, _ = plt.hist(pitchs, bins=18, range = [-90, 90])
    title = 'Pitch distribution - %s pose' % label
    plt.title(title)
    plt.xlabel('angle (degrees)')
    plt.ylabel('number of occurrences')
    plt.savefig(os.path.join(PLOTS, title))

    y = get_y(yaws, yaw_bins, yaw_counts)
    y_all = np.concatenate((y_pain, np.asarray(y).reshape(-1,1)), axis = 1)
    output = Counter([tuple(i) for i in y_all])
    labels = np.vstack(output.keys())
    counts = np.vstack(output.values())
    y = np.vstack([np.where((labels == pain).all(axis = 1))for pain in y_all])
    max_labels = np.max(y)
    for i, index in enumerate(y):
        if counts[index] < 4:
            y[i] = max_labels + 1


    # x is the image name
    # y is the label


    distribution_y =  Counter([tuple(i) for i in y])
    labels = np.hstack(distribution_y.keys())
    counts = np.hstack(distribution_y.values())

    # this is the expected size for each label in the val and test set
    val_size = [ int((c * 0.70)/3) for c in counts]
    test_size = [ int((c * 0.30)) for c in counts]

    desired_size = [val_size, val_size, val_size, test_size]
    final_size = copy.deepcopy(desired_size)


    all_x = [[] for i in range(N_FOLDS + 1)]
    all_y = [[] for i in range(N_FOLDS + 1)]



    class_cross_val =  [[] for i in range(N_FOLDS + 1)] # cross validation for each class (frontal, profile, tilted)
    for i, set_ in enumerate(cross_val):
        indexes_x = [np.where((np.vstack(x) == str(int(id_)) + '.jpg')) for id_ in set_]

        indexes_x = np.hstack([i[0] for i in indexes_x if len(i[0]) != 0])
        all_x[i] = [x[j] for j in indexes_x]
        y_set = [y[j] for j in indexes_x]
        all_y[i] = y_set

        dist_y_set =  Counter([tuple(i) for i in y_set])
        labels_set = np.hstack(dist_y_set.keys())
        counts_set = np.hstack(dist_y_set.values())


        size = final_size[i]
        for j, l in enumerate(labels_set):
            label_index = list(labels).index(l)
            size[j] = max(0, size[j] - 1)
        final_size[i] = size

        class_cross_val[i] = [i for j,i in enumerate(x) if j in indexes_x]
        x = [i for j,i in enumerate(x) if j not in indexes_x]
        y = [i for j,i in enumerate(y) if j not in indexes_x]

    y = np.hstack(y)
    x = np.hstack(x)


    val_size = int((total_images * 0.7) / 3)
    test_size = int((total_images * 0.3))
    #print(label)
    #print('val size : ', val_size)
    #print('test size : ', test_size)
    final_fold_size = [val_size, val_size, val_size, test_size]
    for i, set_ in enumerate(final_size):
        indexes = []
        for j, bin_size in enumerate(set_):
            #print('j: ', j)
            #print('len class: ', len(class_cross_val[i]))
            #print('len final size: ', final_fold_size[i])
            if bin_size > 0 and len(class_cross_val[i]) < final_fold_size[i]:
                remaining = np.max(final_fold_size[i] - len(class_cross_val[i]),0)
                label = labels[j]
                indexes_y = [i for i in range(len(y)) if y[i] == label]

                indexes_to_get = [indexes_y[i] for i in range(min(bin_size, len(indexes_y), remaining))]
                if len(indexes_to_get) > 0:
                    add_x = [x[i] for i in indexes_to_get]
                    x = [ i for j, i in enumerate(x) if j not in indexes_to_get]
                    y = [ i for j, i in enumerate(y) if j not in indexes_to_get]
                    class_cross_val[i] = np.concatenate((class_cross_val[i], np.hstack(add_x)))
                    indexes.append(indexes_y)

        # get a number of x for each label
    print('len x by the end: ', len(x))
    for i, set_ in enumerate(class_cross_val):
        add_i =  int(final_fold_size[i]) - len(set_)
        if add_i > 0 and len(x) > 0:
            class_cross_val[i] = np.concatenate((class_cross_val[i], np.hstack(x[:add_i + 1])))
            x = x[add_i :]

    if len(x) != 0:
        class_cross_val[-1] = np.concatenate((class_cross_val[-1], np.hstack(x)))

    test_angles = [angles[i] for i in range(len(angles)) if angles[i,0] in class_cross_val[-1]]
    with open(os.path.join(ANGLES, '%s_pain_test_angles.pickle' % label), 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(test_angles, f)

    for fold in range(N_FOLDS):
        fold_angles = [angles[i] for i in range(len(angles)) if angles[i,0] in class_cross_val[fold]]
        with open(os.path.join(ANGLES, '%s_pain_val_fold_%d_angles.pickle' % (label, fold)), 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(fold_angles, f)

    train_X = np.concatenate((class_cross_val[0], class_cross_val[1], class_cross_val[2]))
    test_X = class_cross_val[-1]

    return train_X, test_X



def make_copy(list_imgs, subfolder):
    for img in list_imgs:
        shutil.copy2(os.path.join(DATASET, 'images', img), os.path.join(CROSS_VAL, subfolder, img))

#============================================================================
#MAIN
# ==============================================================================

frontal_angles =  pickle.load(open(os.path.join(ANGLES, 'frontal_roll_pitch_yaw.pickle'), "rb"))
tilted_angles =  pickle.load(open(os.path.join(ANGLES, 'tilted_roll_pitch_yaw.pickle'), "rb"))
profile_angles =  pickle.load(open(os.path.join(ANGLES, 'profile_roll_pitch_yaw.pickle'), "rb"))


frontal_train_X,frontal_test_X = get_train_test(frontal_angles, 'frontal')
tilted_train_X,tilted_test_X = get_train_test(tilted_angles, 'tilted')
profile_train_X,profile_test_X = get_train_test(profile_angles, 'profile')


make_copy(frontal_train_X, 'frontal/train')
make_copy(frontal_test_X, 'frontal/test')

make_copy(tilted_train_X, 'tilted/train')
make_copy(tilted_test_X, 'tilted/test')

make_copy(profile_train_X, 'profile/train')
make_copy(profile_test_X, 'profile/test')
