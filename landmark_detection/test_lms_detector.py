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
from train_lms_detector import sorted_image_import
import math
import pickle

DATASET = os.path.join(os.getcwd(),'..','dataset')
#ABS_POSE_FITTER = os.path.join(DATASET,'abs_pose')
RESULTS = os.path.join(os.getcwd(), 'results')
if not os.path.exists(RESULTS):
    os.mkdir(RESULTS)


N_FOLDS = 3
SR = 0.06
ANGLES =  os.path.join(DATASET, '3D_annotations', 'angles')
AUG = 'aug_2'
#MODE = 'final_model'

LMS_SYSTEM = 'absolute' # complete or absolute
if LMS_SYSTEM == 'absolute':
    ABS_POSE = os.path.join(DATASET,'abs_pose')
elif LMS_SYSTEM == 'complete':
    ABS_POSE = os.path.join(DATASET,'abs_pose_complete')


#%%============================================================================
#                       AUXILIAR FUNCTIONS
#==============================================================================
def get_lms_error(plm_errors, label):
    if LMS_SYSTEM == 'absolute':
        if label == 'frontal':
          ear_error = [e for e in np.hstack(plm_errors[:,:6]) if e != 0]
          print('Ears mean error: ', np.mean(ear_error))
          print('Ears success rate: ',len(np.where(np.hstack(ear_error) < SR)[0]) / len(np.hstack(ear_error)))

          nose_error =  [e for e in np.hstack(plm_errors[:,12:24]) if e != 0]
          print('Nose mean error: ', np.mean(nose_error))
          print('Nose success rate: ',len(np.where(np.hstack(nose_error) < SR)[0]) / len(np.hstack(nose_error)))

          eye_error = [e for e in np.hstack(plm_errors[:,6:12]) if e != 0]
          print('Eye mean error: ', np.mean(eye_error))
          print('Eye success rate: ',len(np.where(np.hstack(eye_error) < SR)[0]) / len(np.hstack(eye_error)))

          r_eye_error =  [e for e in np.hstack(plm_errors[:,24:30]) if e != 0]
          print('Second eye mean error: ', np.mean(r_eye_error))
          print('Second eye success rate: ',len(np.where(np.hstack(r_eye_error) < SR)[0]) / len(np.hstack(eye_error)))

          return [np.mean(ear_error), np.mean(nose_error), np.mean(eye_error), np.mean(r_eye_error)], np.mean(np.hstack([ear_error,nose_error, eye_error, r_eye_error]))
          #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        elif label == 'tilted':
          ear_error =  [e for e in   np.hstack(plm_errors[:,:6]) if e != 0]
          print('Ears mean error: ', np.mean(ear_error))
          print('Ears success rate: ',len(np.where(np.hstack(ear_error) < SR)[0]) / len(np.hstack(ear_error)))

          nose_error = [e for e in  np.hstack(plm_errors[:,12:18]) if e != 0]
          print('Nose mean error: ', np.mean(nose_error))
          print('Nose success rate: ',len(np.where(np.hstack(nose_error) < SR)[0]) / len(np.hstack(nose_error)))

          eye_error = [e for e in  np.hstack(plm_errors[:,6:12]) if e != 0]
          print('Eye mean error: ', np.mean(eye_error))
          print('Eye success rate: ',len(np.where(np.hstack(eye_error) < SR)[0]) / len(np.hstack(eye_error)))

          mouth_error =  [e for e in  np.hstack(plm_errors[:,25:28]) if e != 0]
          print('Mouth mean error: ', np.mean(mouth_error))
          print('Mouth success rate: ',len(np.where(np.hstack(mouth_error) < SR)[0]) / len(np.hstack(mouth_error)))

          return [np.mean(ear_error), np.mean(nose_error), np.mean(eye_error), np.mean(mouth_error)], np.mean(np.hstack([ear_error, nose_error, eye_error, mouth_error]))
          #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        elif label == 'profile':
          ear_error = [e for e in np.hstack(plm_errors[:,:3]) if e != 0]
          print('Ears mean error: ', np.mean(ear_error))
          print('Ears success rate: ',len(np.where(np.hstack(ear_error) < SR)[0]) / len(np.hstack(ear_error)))

          nose_error = [e for e in np.hstack(plm_errors[:,11:17]) if e != 0]
          print('Nose mean error: ', np.mean(nose_error))
          print('Nose success rate: ',len(np.where(np.hstack(nose_error) < SR)[0]) / len(np.hstack(nose_error)))

          eye_error = [e for e in np.hstack(plm_errors[:,17:23]) if e != 0]
          print('Eye mean error: ', np.mean(eye_error))
          print('Eye success rate: ',len(np.where(np.hstack(eye_error) < SR)[0]) / len(np.hstack(eye_error)))

          mouth_error = [e for e in  np.hstack(plm_errors[:,7:11]) if e != 0]
          print('Mouth mean error: ', np.mean(mouth_error))
          print('Mouth success rate: ',len(np.where(np.hstack(mouth_error) < SR)[0]) / len(np.hstack(mouth_error)))

          cheek_error =  [e for e in  np.hstack(plm_errors[:,3:7]) if e != 0]
          print('Cheek mean error: ', np.mean(cheek_error))
          print('Cheek success rate: ',len(np.where(np.hstack(cheek_error) < SR)[0]) / len(np.hstack(cheek_error)))

          return [np.mean(ear_error), np.mean(nose_error), np.mean(eye_error), np.mean(mouth_error), np.mean(cheek_error)], np.mean(np.hstack([ear_error, nose_error, eye_error, mouth_error]))


    elif LMS_SYSTEM == 'complete':
        if label == 'frontal':
            ear_error = [e for e in np.hstack(plm_errors[:,:11]) if e != 0]
            print('Ears mean error: ', np.mean(ear_error))
            print('Ears success rate: ',len(np.where(np.hstack(ear_error) < SR)[0]) / len(np.hstack(ear_error)))

            nose_error =  [e for e in np.hstack(plm_errors[:,16:29]) if e != 0]
            print('Nose mean error: ', np.mean(nose_error))
            print('Nose success rate: ',len(np.where(np.hstack(nose_error) < SR)[0]) / len(np.hstack(nose_error)))

            eye_error = [e for e in np.hstack(plm_errors[:,10:17]) if e != 0]
            print('Eye mean error: ', np.mean(eye_error))
            print('Eye success rate: ',len(np.where(np.hstack(eye_error) < SR)[0]) / len(np.hstack(eye_error)))

            r_eye_error =  [e for e in np.hstack(plm_errors[:,28:35]) if e != 0]
            print('Second eye mean error: ', np.mean(r_eye_error))
            print('Second eye success rate: ',len(np.where(np.hstack(r_eye_error) < SR)[0]) / len(np.hstack(eye_error)))

            return [np.mean(ear_error), np.mean(nose_error), np.mean(eye_error), np.mean(r_eye_error)], np.mean(np.hstack([ear_error,nose_error, eye_error, r_eye_error]))

        elif label == 'tilted':
            ear_error =  [e for e in   np.hstack(plm_errors[:,:11]) if e != 0]
            print('Ears mean error: ', np.mean(ear_error))
            print('Ears success rate: ',len(np.where(np.hstack(ear_error) < SR)[0]) / len(np.hstack(ear_error)))

            nose_error = [e for e in  np.hstack(plm_errors[:,16:23]) if e != 0]
            print('Nose mean error: ', np.mean(nose_error))
            print('Nose success rate: ',len(np.where(np.hstack(nose_error) < SR)[0]) / len(np.hstack(nose_error)))

            eye_error = [e for e in  np.hstack(plm_errors[:,10:17]) if e != 0]
            print('Eye mean error: ', np.mean(eye_error))
            print('Eye success rate: ',len(np.where(np.hstack(eye_error) < SR)[0]) / len(np.hstack(eye_error)))

            mouth_error =  [e for e in  np.hstack(plm_errors[:,29:32]) if e != 0]
            print('Mouth mean error: ', np.mean(mouth_error))
            print('Mouth success rate: ',len(np.where(np.hstack(mouth_error) < SR)[0]) / len(np.hstack(mouth_error)))

            return [np.mean(ear_error), np.mean(nose_error), np.mean(eye_error), np.mean(mouth_error)], np.mean(np.hstack([ear_error, nose_error, eye_error, mouth_error]))

        elif label == 'profile':
            ear_error = [e for e in np.hstack(plm_errors[:,:8]) if e != 0]
            print('Ears mean error: ', np.mean(ear_error))
            print('Ears success rate: ',len(np.where(np.hstack(ear_error) < SR)[0]) / len(np.hstack(ear_error)))

            nose_error = [e for e in np.hstack(plm_errors[:,16:23]) if e != 0]
            print('Nose mean error: ', np.mean(nose_error))
            print('Nose success rate: ',len(np.where(np.hstack(nose_error) < SR)[0]) / len(np.hstack(nose_error)))

            eye_error = [e for e in np.hstack(plm_errors[:,22:29]) if e != 0]
            print('Eye mean error: ', np.mean(eye_error))
            print('Eye success rate: ',len(np.where(np.hstack(eye_error) < SR)[0]) / len(np.hstack(eye_error)))

            mouth_error = [e for e in  np.hstack(plm_errors[:,12:17]) if e != 0]
            print('Mouth mean error: ', np.mean(mouth_error))
            print('Mouth success rate: ',len(np.where(np.hstack(mouth_error) < SR)[0]) / len(np.hstack(mouth_error)))

            return [np.mean(ear_error), np.mean(nose_error), np.mean(eye_error), np.mean(mouth_error)], np.mean(np.hstack([ear_error, nose_error, eye_error, mouth_error]))


#==============================================================================
def test_eval(fitter, images, files, mean = False, pose ='',  verbose=True, save_images = False, folder = DATASET, fold = ''):
    if not os.path.exists(folder):
      os.mkdir(folder)

    error = 0
    errors = []
    plm_errors = []

    all_errors = []
    for i, image in enumerate(images):

      #print(files[i])
      lms_gt = image.landmarks['PTS'].lms.as_vector().reshape((-1, 2))
      if verbose:
        #print('Tested', i+1, ' of ', len(images), '\n')
        pass
      if mean:
          lms_pred = fitter.lms.as_vector().reshape((-1, 2))
          """
          if save_images:
              open_cv_frame = image.as_PILImage().convert('RGB')
              open_cv_frame = np.array(open_cv_frame)
              j = 0
              r = 3
              for (y, x) in lms_pred:
                  if j in lms:
                      cv.circle(open_cv_frame, (int(x), int(y)), r, (255,0,0), thickness=-1, lineType=cv.LINE_AA)
                  else:
                      cv.circle(open_cv_frame, (int(x), int(y)), r, (255,0,255), thickness=-1, lineType=cv.LINE_AA)
                      j += 1
              cv.imwrite(os.path.join(folder, '%d.jpg' % i), cv.cvtColor(open_cv_frame, cv.COLOR_BGR2RGB))
           """

      else:

          result = fitter.fit_from_bb(image = image, bounding_box =
                                      image.landmarks['PTS'].bounding_box(),
                                      gt_shape=image.landmarks['PTS'])

          lms_pred = result.final_shape.as_vector().reshape((-1, 2))


      open_cv_frame = image.as_PILImage().convert('RGB')
      open_cv_frame = np.array(open_cv_frame)

      h, w = np.shape(open_cv_frame)[:2]

      # Ground truth point is out of the frame
      index_to_exclude = []
      for p in range(len(lms_gt)):
        if lms_gt[p,0] >= h or lms_gt[p,1] >= w or lms_gt[p,1] <= 0 or lms_gt[p,0] <= 0:
          index_to_exclude.append(p)

      # Mempo works with (y,x) and not (x,y)!
      if LMS_SYSTEM == 'absolute':
          if label == 'frontal':
            eye_center = [(lms_gt[9,1] + lms_gt[6,1])/2, (lms_gt[8,0] + lms_gt[11,0])/2]
            nose_center = [(lms_gt[12,1] + lms_gt[14,1])/2, (lms_gt[16,0] + lms_gt[13,0])/2]

            distance_1 = list(np.array(eye_center) - np.array(nose_center))
            distance_1 = [abs(i) for i in distance_1]
            yard_stick_distance_1 = math.sqrt(distance_1[0] **2 + distance_1[1] ** 2)
            eye_2_center = [(lms_gt[26,1] + lms_gt[29,1])/2, (lms_gt[24,0] + lms_gt[27,0])/2]
            nose_2_center = [(lms_gt[17,1] + lms_gt[22,1])/2, (lms_gt[20,0] + lms_gt[18,0])/2]
            distance_2 = list(np.array(eye_2_center) - np.array(nose_2_center))
            distance_2 = [abs(i) for i in distance_2]
            yard_stick_distance_2 = math.sqrt(distance_2[0] **2 + distance_2[1] ** 2)
            norm = np.mean([yard_stick_distance_1,yard_stick_distance_2], axis = 0)

          elif label == 'tilted':
            eye_center = [(lms_gt[9,1] + lms_gt[6,1])/2, (lms_gt[8,0] + lms_gt[11,0])/2]
            #select the nose center
            nose_center = [(lms_gt[14,1] + lms_gt[12,1])/2, (lms_gt[16,0] + lms_gt[13,0])/2]
            #calculate the distance
            yard_stick = list(np.array(eye_center) - np.array(nose_center))
            norm = math.sqrt(yard_stick[0] **2 + yard_stick[1] ** 2)

          elif label == 'profile':
            eye_center = [(lms_gt[17,1] + lms_gt[20,1])/2, (lms_gt[22,0] + lms_gt[19,0])/2]
            nose_center = [(lms_gt[12,1] + lms_gt[15,1])/2, (lms_gt[11,0] + lms_gt[13,0])/2]
            yard_stick = list(np.array(eye_center) - np.array(nose_center))
            norm = math.sqrt(yard_stick[0] **2 + yard_stick[1] ** 2)

          #eye_center = [eye_center[-1], eye_center[0]]
          #nose_center = [nose_center[-1], nose_center[0]]
          #cv.line(open_cv_frame, tuple([int(e) for e in eye_center]), tuple([int(n) for n in nose_center]), (255,255,255), thickness=3)

          errors_image = []
          for k in range(len(lms_pred)):
            if k not in index_to_exclude:
              e = np.sqrt((lms_pred[k,0] - lms_gt[k,0]) ** 2 + (lms_pred[k,1] - lms_gt[k,1]) ** 2) / norm
              error += e
              errors_image.append(e)
            else:
              e = 0
              errors_image.append(e)
          plm_errors.append(errors_image)

      elif LMS_SYSTEM == 'complete':
          if label == 'frontal':
              eye_center = [(lms_gt[13,1] + lms_gt[10,1])/2, (lms_gt[12,0] + lms_gt[15,0])/2]
              nose_center = [(lms_gt[16,1] + lms_gt[18,1])/2, (lms_gt[20,0] + lms_gt[17,0])/2]

              distance_1 = list(np.array(eye_center) - np.array(nose_center))
              distance_1 = [abs(i) for i in distance_1]
              yard_stick_distance_1 = math.sqrt(distance_1[0] **2 + distance_1[1] ** 2)
              eye_2_center = [(lms_gt[30,1] + lms_gt[33,1])/2, (lms_gt[28,0] + lms_gt[31,0])/2]
              nose_2_center = [(lms_gt[23,1] + lms_gt[26,1])/2, (lms_gt[24,0] + lms_gt[22,0])/2]
              distance_2 = list(np.array(eye_2_center) - np.array(nose_2_center))
              distance_2 = [abs(i) for i in distance_2]
              yard_stick_distance_2 = math.sqrt(distance_2[0] **2 + distance_2[1] ** 2)
              norm = np.mean([yard_stick_distance_1,yard_stick_distance_2], axis = 0)

          elif label == 'tilted':
              eye_center = [(lms_gt[13,1] + lms_gt[10,1])/2, (lms_gt[12,0] + lms_gt[15,0])/2]
              #select the nose center
              nose_center = [(lms_gt[18,1] + lms_gt[16,1])/2, (lms_gt[20,0] + lms_gt[17,0])/2]
              #calculate the distance
              yard_stick = list(np.array(eye_center) - np.array(nose_center))
              norm = math.sqrt(yard_stick[0] **2 + yard_stick[1] ** 2)

          elif label == 'profile':
              eye_center = [(lms_gt[22,1] + lms_gt[25,1])/2, (lms_gt[27,0] + lms_gt[24,0])/2]
              nose_center = [(lms_gt[17,1] + lms_gt[20,1])/2, (lms_gt[16,0] + lms_gt[18,0])/2]
              yard_stick = list(np.array(eye_center) - np.array(nose_center))
              norm = math.sqrt(yard_stick[0] **2 + yard_stick[1] ** 2)

          #eye_center = [eye_center[-1], eye_center[0]]
          #nose_center = [nose_center[-1], nose_center[0]]
          #cv.line(open_cv_frame, tuple([int(e) for e in eye_center]), tuple([int(n) for n in nose_center]), (255,255,255), thickness=3)

          errors_image = []
          for k in range(len(lms_pred)):
              if k not in index_to_exclude:
                  e = np.sqrt((lms_pred[k,0] - lms_gt[k,0]) ** 2 + (lms_pred[k,1] - lms_gt[k,1]) ** 2) / norm
                  error += e
                  errors_image.append(e)
              else:
                  e = 0
                  errors_image.append(e)
          plm_errors.append(errors_image)

      r = 3
      file = files[i].split('/')[-1].split('.')[0]

      if save_images:
        gt_frame = open_cv_frame.copy()

        j = 0
        for (y, x) in lms_pred:
          cv.circle(open_cv_frame, (int(x), int(y)), r, (255,0,0), thickness=-1, lineType=cv.LINE_AA)
          j += 1

        file = files[i].split('/')[-1].split('.')[0]
        cv.imwrite(os.path.join(folder, '%s_%.3f.jpg' % (file,np.mean(plm_errors[-1]))), cv.cvtColor(open_cv_frame, cv.COLOR_BGR2RGB))
        #print(os.path.join(folder, '%s_%.3f_gt.jpg' % (file,np.mean(plm_errors[-1]))))
        for k in range(len(lms_pred)):
          cv.line(open_cv_frame, (int(lms_pred[k, 1]), int(lms_pred[k, 0])), (int(lms_gt[k, 1]), int(lms_gt[k, 0])), (255,255,255), thickness=2)

        cv.imwrite(os.path.join(folder, '%s_%.3f_lines.jpg' % (file,np.mean(plm_errors[-1]))), cv.cvtColor(open_cv_frame, cv.COLOR_BGR2RGB))


        j = 0
        for (y, x) in lms_gt:
          cv.circle(gt_frame, (int(x), int(y)), r, (255,0,0), thickness=-1, lineType=cv.LINE_AA)
          j += 1

        cv.imwrite(os.path.join(folder, '%s_gt.jpg' % file), cv.cvtColor(gt_frame, cv.COLOR_BGR2RGB))

          #print(os.path.join(folder, '%s_%.3f_gt.jpg' % (file,np.mean(plm_errors[-1]))))
        for k in range(len(lms_pred)):
          cv.line(open_cv_frame, (int(lms_pred[k, 1]), int(lms_pred[k, 0])), (int(lms_gt[k, 1]), int(lms_gt[k, 0])), (255,255,255), thickness=2)

        #cv.imwrite(os.path.join(folder, '%s_%.3f.jpg' % (file,np.mean(plm_errors[-1]))), cv.cvtColor(gt_frame, cv.COLOR_BGR2RGB))

      all_errors.append([file, errors_image])

    if fold != -1:
        with open(os.path.join(RESULTS,'results_%s_%d_pert_%d_%s.pickle' % (LMS_SYSTEM, fold, pert, label)), 'wb') as f:
            #with open(os.path.join(RESULTS,'full_%s_results_%s.pickle' % (AUG, label)), 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            print('print results: ', len(all_errors))
            pickle.dump(all_errors, f)
    else:
        with open(os.path.join(RESULTS,'results_%s_final_pert_%d_%s.pickle' % (LMS_SYSTEM,pert, label)), 'wb') as f:
            #with open(os.path.join(RESULTS,'full_%s_results_%s.pickle' % (AUG, label)), 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            print('print results: ', len(all_errors))
            pickle.dump(all_errors, f)

    plm_errors = np.vstack(plm_errors)
    per_roi_errors, mean_roi_error = get_lms_error(plm_errors, label)

    if fold != -1:
        with open(os.path.join(RESULTS,'results_per_roi_%s_%d_pert_%d_%s.pickle' % (LMS_SYSTEM, fold, pert, label)), 'wb') as f:
            #with open(os.path.join(RESULTS,'full_%s_results_per_roi_%s.pickle' % (AUG, label)), 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(per_roi_errors, f)
    else:
        with open(os.path.join(RESULTS,'results_per_roi_%s_pert_%d_final_%s.pickle' % (LMS_SYSTEM,pert, label)), 'wb') as f:
            #with open(os.path.join(RESULTS,'full_%s_results_per_roi_%s.pickle' % (AUG, label)), 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(per_roi_errors, f)

    print('Mean error:     ', np.mean([e for e in np.hstack(plm_errors) if e != 0]))
    print('Mean roi error:     ', mean_roi_error)
    print('Less than 6% error:', len(np.where(np.hstack([e for e in np.hstack(plm_errors) if e != 0]) < SR) [0]) / len(np.hstack([e for e in np.hstack(plm_errors) if e != 0])))

    return plm_errors

def get_test_results(path_to_images, fitter_path, save_images_folder, cross_val = False, fold = 0, mean = False):
    images, files = sorted_image_import(path_to_images)
    if cross_val == True:
        val = np.vstack(pickle.load(open(os.path.join(ANGLES, '%s_pain_val_fold_%d_angles.pickle' % (save_images_folder.split('/')[-1], fold)), 'rb')))
        indexes_val = [i for i in range(len(files)) if files[i].split('/')[-1].split('.')[0] + '.jpg' in val[:,0]]
        file_list = [files[i] for i in indexes_val]
        print('len validation set: ', len(file_list))
        images = LazyList([partial(mio.import_image,f) for f in file_list])
        landmarks = images.map(lambda x: x.landmarks) #Extracts the landmarks (associated with each image)
        fitter = mio.import_pickle(fitter_path)
        errors = test_eval(fitter, images, files, mean = mean,  folder = save_images_folder, save_images = True, fold = fold)

    else:
        path_to_images = os.path.join(ABS_POSE,label,'test/')
        images, files = sorted_image_import(path_to_images)
        fitter = mio.import_pickle(fitter_path)
        errors = test_eval(fitter, images, files, mean = mean, folder = save_images_folder, save_images = True, fold = fold)

#%%============================================================================
#                                   MAIN
#==============================================================================
"""
cross_val = True
AUG = 1.5
data_aug = True
print(LMS_SYSTEM,  '\n====================')
if cross_val:
    #for label in ['frontal', 'tilted', 'profile']:
    #for label in ['tilted']:
    #for label, pert in [['frontal', 90], ['tilted', 100],['profile', 100]]:
    for label, pert in [['frontal', 90], ['tilted', 100],['profile', 100]]:
    #for label, pert in [['profile', 100]]:
        print(label, '\n====================')
        print('PERT = ', pert)
        for p in ['ert']: #'ert_' + prefix + '_pert_%d' %n_pert)
            print(p, '\n----------------------------')
            for fold in range(N_FOLDS):
                if data_aug:
                    prefix = '%s_' % p + 'fold_' + str(fold) + '_pert_%d_aug_%.1f.pkl' % (pert,AUG)
                elif data_aug == False:
                    prefix = '%s_' % p + 'fold_' + str(fold) + '_pert_%d.pkl' % (pert)
                get_test_results(os.path.join(ABS_POSE,label, 'train'),
                                             os.path.join(ABS_POSE, label,'train', 'fitters', prefix),
                                             os.path.join(os.getcwd(), label), cross_val = cross_val, fold = fold, mean = False)
                print('.........................')

else:

    for label, pert in [['frontal', 90], ['tilted', 100],['profile', 100]]:
        print(label, '\n====================')
        print('PERT = ', pert)
        for p in ['ert', 'sdm']: #'ert_' + prefix + '_pert_%d' %n_pert)
            print(p, '\n----------------------------')
            if data_aug:
                prefix = '%s_' % p + 'final_%s_pert_%d_aug_%.1f.pkl' % (label, pert, AUG)
            elif data_aug == False:
                prefix = '%s_' % p + 'final_%s_pert_%d.pkl' % (label, pert)

            get_test_results(os.path.join(ABS_POSE,label, 'train'),
                                         os.path.join(ABS_POSE, label,'train', 'fitters', prefix),
                                         os.path.join(os.getcwd(), label), cross_val = False, fold = 0, mean = False)

        p = 'mean'
        print(p, '\n----------------------------')
        prefix = '%s_' % p + 'final_' + label + label + '_pert_%d.pkl' % pert
        get_test_results(os.path.join(ABS_POSE,label, 'train'),
                         os.path.join(ABS_POSE, label,'train', 'fitters', prefix),
                         os.path.join(os.getcwd(), label), cross_val = False, fold = 3, mean = True)
"""