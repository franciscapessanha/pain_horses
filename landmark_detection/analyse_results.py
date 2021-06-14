#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 14:44:35 2020

@author: franciscapessanha
"""

import pickle
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from test_lms_detector import get_lms_error
import random

#%%
DATASET = os.path.join(os.getcwd(), '..', 'dataset')
RESULTS = os.path.join(os.getcwd(), 'results')
ANGLES =  os.path.join(DATASET, '3D_annotations', 'angles')
PLOTS = os.path.join(os.getcwd(), 'plots')
if not os.path.exists(PLOTS):
    os.mkdir(PLOTS)

mode = 'cross_val'
#%%

def get_error_per_pose(angles, results,title, angle, index, train):

    average_results = [[i[0], np.mean(i[1])] for i in results]
    average_results.sort()
    angles = angles[angles[:,0].argsort()]

    abs_train_angle =  np.asarray([abs(float(y)) for y in train[:,index]])
    abs_val = np.asarray([abs(float(y)) for y in angles[:,index]])
    bins = [i for i in range(0,95, 5)]
    counts, bins, _ = plt.hist(abs_train_angle, bins=len(bins), range=(0,95))


    print('lenght abs_angles (validation): ', len(abs_val))
    print('lenght results: ', len(results))

    bin_means, bin_edges, _  = stats.binned_statistic(abs_val, np.vstack(average_results)[:,1].astype(np.float), bins = bins)
    #
    #print('bin_means: ', bin_means)
    #print('bin_edges: ', bin_edges)

    plt.figure()
    plt.bar(bin_edges[1:], bin_means, width = bin_edges[1] - bin_edges[0] - 0.1)
    plt.xlabel('Absolute %s' % angle)
    plt.ylabel('Mean Error')
    title = title + ' (%s angle)' % angle
    plt.title(title)
    plt.savefig(os.path.join(PLOTS, title))

    #return bin_means, bin_edges, counts,  np.median([np.mean(i[1]) for i in results], axis = 0)
    return bin_means, bin_edges, counts,  np.median([np.min(i[1]) for i in results], axis = 0)

def get_final_error_per_pose(angles, mean_bin,title, angle, index, train, results):

    average_results = [[i[0], np.mean(i[1])] for i in results]
    average_results.sort()
    angles = angles[angles[:,0].argsort()]

    abs_train_angle =  np.asarray([abs(float(y)) for y in train[:,index]])
    abs_angle = np.asarray([abs(float(y)) for y in angles[:,index]])
    bins = [i for i in range(0,95, 5)]
    counts, bins, _ = plt.hist(abs_train_angle, bins=len(bins), range=(0,95))


    bin_means, bin_edges, _  = stats.binned_statistic(abs_angle, np.vstack(average_results)[:,1].astype(np.float), bins = bins)


    return bin_means, bin_edges, counts,  np.mean([np.min(i[1]) for i in results], axis = 0)


def get_results_folds(label, pert,  N_FOLDS = 3):
    angles_train = np.vstack(np.load(open(os.path.join(ANGLES, '%s_pain_train_angles.pickle' % label), 'rb'), allow_pickle = True))
    if mode == 'cross_val':
        for k in range(N_FOLDS):
            all_means = []
            all_edges = []
            all_min_errors = []
            for index, angle in [[-1, 'yaw'], [-2, 'pitch'], [-3, 'roll']]:
                results = np.load(open(os.path.join(RESULTS, 'results_%s_%d_pert_%d_%s.pickle' % (LMS_SYSTEM, k, pert, label)), 'rb'), allow_pickle = True)
                angles = np.vstack(np.load(open(os.path.join(ANGLES, '%s_pain_val_fold_%d_angles.pickle' % (label, k)), 'rb'), allow_pickle = True))
                train = np.vstack([a for a in angles_train if a not in angles])
                bin_means, bin_edges, counts, min_error = get_error_per_pose(angles, results, ' %s - fold %d' % (label, k), angle, index, train)
                all_means.append(bin_means)
                all_edges.append(bin_edges)
                all_min_errors.append(min_error)


            random_values = []
            number_values_yaw = 0
            yaw_means = all_means[0]
            yaw_edges = all_edges[0]
            for i, mean in enumerate(yaw_means):
                if not np.isnan(mean) and mean != 0:
                    aug_factor = (mean/min_error)** alpha
                    if aug_factor > 1:
                        n_new_values = int(counts[i] * (aug_factor - 1))
                        number_values_yaw += n_new_values
                        for j in range(n_new_values):
                            random_values.append(random.uniform(yaw_edges[i], yaw_edges[i + 1]))

            with open(os.path.join(os.getcwd(), 'data_aug', 'error3_%s_%s_%s_fold_%d.pickle' % (label, AUG, 'yaw', k)), 'wb') as f:
            #with open(os.path.join(os.getcwd(),'%s_%s_fold_%d.pickle' % (label, angle, k)), 'wb') as f:
                  # Pickle the 'data' dictionary using the highest protocol available.
                      pickle.dump(random_values, f)

            for i, angle in [[1, 'pitch'], [2, 'roll']]:
                n_new_values = []
                number_values = 0
                random_values = []
                bin_edges = all_edges[i]
                for j, mean in enumerate(all_means[i]):
                    if not np.isnan(mean) and mean != 0:
                        aug_factor = ((mean/all_min_errors[0])** alpha)
                        if aug_factor > 0:
                            n_new_values.append(int(counts[i] * (aug_factor) -1))
                            number_values += n_new_values[-1]
                        else:
                            n_new_values.append(0)
                    else:
                        n_new_values.append(0)

                print(n_new_values)
                for j, mean in enumerate(all_means[i]):
                    N = int(np.ceil(n_new_values[j] * (number_values_yaw/number_values)))
                    for n in range(N):
                        random_values.append(random.uniform(bin_edges[i], bin_edges[i + 1]))
                print('len(random): ', len(random_values))
                print('len yaw: ', number_values_yaw)
                with open(os.path.join(os.getcwd(), 'data_aug', 'error3_%s_%s_%s_fold_%d.pickle' % (label, AUG, angle, k)), 'wb') as f:
                    #with open(os.path.join(os.getcwd(),'%s_%s_fold_%d.pickle' % (label, angle, k)), 'wb') as f:
                        # Pickle the 'data' dictionary using the highest protocol available.
                      pickle.dump(random_values, f)

    else:
            all_means = []
            all_edges = []
            all_min_errors = []
            for index, angle in [[-1, 'yaw'], [-2, 'pitch'], [-3, 'roll']]:
                all_m = []
                all_e = []
                all_min = []
                for k in range(N_FOLDS):

                    results = np.load(open(os.path.join(RESULTS, 'results_%s_%d_pert_%d_%s.pickle' % (LMS_SYSTEM, k, pert, label)), 'rb'), allow_pickle = True)
                    angles = np.vstack(np.load(open(os.path.join(ANGLES, '%s_pain_val_fold_%d_angles.pickle' % (label, k)), 'rb'), allow_pickle = True))
                    train = np.vstack([a for a in angles_train if a not in angles])
                    bin_means, bin_edges, counts, min_error = get_error_per_pose(angles, results, ' %s - fold %d' % (label, k), angle, index, train)

                    all_m.append(bin_means)
                    all_e.append(bin_edges)
                    all_min.append(min_error)
                all_m = np.vstack(all_m)
                all_e = np.vstack(all_e)
                all_min = np.vstack(all_min)
                all_means.append(np.mean(all_m, axis = 0))
                all_edges.append(np.mean(all_e, axis = 0))
                all_min_errors.append(np.mean(all_min, axis = 0))

            random_values = []
            number_values_yaw = 0
            yaw_means = all_means[0]
            yaw_edges = all_edges[0]
            for i, mean in enumerate(yaw_means):
                if not np.isnan(mean) and mean != 0:
                    aug_factor = (mean/min_error)** alpha
                    if aug_factor > 1:
                        n_new_values = int(counts[i] * (aug_factor - 1))
                        number_values_yaw += n_new_values
                        for j in range(n_new_values):
                            random_values.append(random.uniform(yaw_edges[i], yaw_edges[i + 1]))

            with open(os.path.join(os.getcwd(), 'data_aug', 'error3_%s_%s_%s_final.pickle' % (label, AUG, 'yaw')), 'wb') as f:
            #with open(os.path.join(os.getcwd(),'%s_%s_fold_%d.pickle' % (label, angle, k)), 'wb') as f:
                  # Pickle the 'data' dictionary using the highest protocol available.
                      pickle.dump(random_values, f)

            for i, angle in [[1, 'pitch'], [2, 'roll']]:
                n_new_values = []
                number_values = 0
                random_values = []
                bin_edges = all_edges[i]
                for j, mean in enumerate(all_means[i]):
                    if not np.isnan(mean) and mean != 0:
                        aug_factor = ((mean/all_min_errors[0])** alpha)
                        if aug_factor > 0:
                            n_new_values.append(int(counts[i] * (aug_factor) -1))
                            number_values += n_new_values[-1]
                        else:
                            n_new_values.append(0)
                    else:
                        n_new_values.append(0)

                print(n_new_values)
                for j, mean in enumerate(all_means[i]):
                    N = int(np.ceil(n_new_values[j] * (number_values_yaw/number_values)))
                    for n in range(N):
                        random_values.append(random.uniform(bin_edges[i], bin_edges[i + 1]))
                print('len(random): ', len(random_values))
                print('len yaw: ', number_values_yaw)
                print(os.path.join(os.getcwd(), 'data_aug', 'error3_%s_%s_%s_final.pickle' % (label, AUG, angle)))
                with open(os.path.join(os.getcwd(), 'data_aug', 'error3_%s_%s_%s_final.pickle' % (label, AUG, angle)), 'wb') as f:
                    #with open(os.path.join(os.getcwd(),'%s_%s_fold_%d.pickle' % (label, angle, k)), 'wb') as f:
                        # Pickle the 'data' dictionary using the highest protocol available.
                      pickle.dump(random_values, f)

def get_results_per_ROI(label, pert, N_FOLDS = 3):
    all_errors = []
    mean_error = []
    for k in range(N_FOLDS):
        results = np.load(open(os.path.join(RESULTS, 'results_per_roi_%s_%d_pert_%d_%s.pickle' % (LMS_SYSTEM, k, pert, label)), 'rb'), allow_pickle = True)
        #results = np.load(open(os.path.join(RESULTS, '%d_%s_results_%s.pickle' % (k, AUG, label)), 'rb'), allow_pickle = True)
        results = np.asarray([i[1] for i in results]).astype(np.float)
        mean_error.append(np.mean(results))
        all_errors.append(get_lms_error(results, label))

    if label == 'frontal':
        keywords = ['Ears', 'Nose', 'Left Eye', 'Right Eye']

    if label == 'tilted':
        keywords = ['Ears', 'Nose', 'Left Eye', 'Mouth']

    if label == 'profile':
        keywords = ['Ears', 'Nose', 'Left Eye', 'Mouth', 'Cheek']

    print('=============================================================')
    print('Mean Error: ', np.mean(mean_error), ' +/- ', np.std(mean_error))
    lms_mean_error = np.mean(np.asarray(all_errors), axis = 0)
    lms_std = np.std(all_errors, axis = 0)
    for i, word in enumerate(keywords):
        print('%s Error: ' % word, np.mean(lms_mean_error[i]), ' +/- ', lms_std[i])

    return all_errors

#%%

LMS_SYSTEM = 'absolute'
mode = 'final_model'
for alpha in [0.1]:
    print('%.2f\n==================' % alpha)
    AUG = 'aug_%.2f' % alpha
    #get_results_folds('frontal', 90)
    get_results_folds('tilted', 100)
    #get_results_folds('profile', 100)


#frontal_errors =  get_results_per_ROI('frontal')
#tilted_errors =  get_results_per_ROI('tilted')
#profile_errors =  get_results_per_ROI('profile')

