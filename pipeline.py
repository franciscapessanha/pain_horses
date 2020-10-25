#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 13:28:52 2020

@author: franciscapessanha
"""

#%%============================================================================
#                       IMPORTS AND INITIALIZATIONS
#============================================================================

import numpy as np
import os
import glob
import cv2 as cv
import pickle
from pose_classifier import get_HOGs
import menpo.io as mio
import menpo
import subprocess
import argparse
import json
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import time
import math
from ast import literal_eval
from horse_classifier import get_rounded
import pickle
from fusion import mse_classifictaion_fusion

# from pain_estimation import extract_features_prototype, function_prototype, predict_total_score, k_nearest_confidence_score_extraction
from HOG import extract_features_prototype
from statistics import mean
from sklearn.metrics import mean_squared_error

DATASET = os.path.join(os.getcwd(), 'dataset')
ABS_POSE = os.path.join(DATASET,'abs_pose')
MODELS = os.path.join(os.getcwd(), 'models')

TEST = os.path.join(os.getcwd(), 'test')

if not os.path.exists(TEST):
    os.mkdir(TEST)

detection_time = []
pose_time = []
lms_time = []
pain_time = []

filename = os.path.join(DATASET, 'pain_annotations.xlsx')
pain_gt = pd.read_excel(filename)

data = pickle.load(open( os.path.join(DATASET, 'lms_annotations.pkl'), "rb" ))

#%%============================================================================
#                           AUXILIAR FUNCTIONS
#============================================================================

test_0 = glob.glob(os.path.join(ABS_POSE,'0_30/test/','*.jpg'))
test_30 = glob.glob(os.path.join(ABS_POSE,'30_60/test/','*.jpg'))
test_60 = glob.glob(os.path.join(ABS_POSE,'60_90/test/','*.jpg'))
test = np.concatenate((test_0, test_30, test_60))

test_indexes = []
for path in test:
	index =  int(path.split('/')[-1].split('.')[0])
	test_indexes.append(index)

indexes_0 = []
for path in test_0:
    index =  int(path.split('/')[-1].split('.')[0])
    indexes_0.append(index)

indexes_30 = []
for path in test_30:
    index =  int(path.split('/')[-1].split('.')[0])
    indexes_30.append(index)

indexes_60 = []
for path in test_60:
    index =  int(path.split('/')[-1].split('.')[0])
    indexes_60.append(index)
#%%
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def load_models(): #Models should all be in a models folder in the same directory
    #pose_classifier = pickle.load(open(os.path.join(MODELS, 'pose_classifier.sav'), 'rb'))
    pose_classifier = []

    fitters = []
    fitter_path = os.path.join(MODELS, 'ert_frontal.pkl')
    fitters.append(mio.import_pickle(fitter_path))

    fitter_path = os.path.join(MODELS, 'ert_tilted.pkl')
    fitters.append(mio.import_pickle(fitter_path))

    fitter_path = os.path.join(MODELS, 'ert_profile.pkl')
    fitters.append(mio.import_pickle(fitter_path))

    return pose_classifier, fitters

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def detect_face(image):
    output = subprocess.run(['./darknet', 'detector', 'test', 'data/obj.data', 'models/yolov3-horse.cfg', 'models/yolov3-horse_4000.weights',  '-dont_show', '-ext_output', image], stdout=subprocess.PIPE)
    lines = output.stdout.splitlines()[-1]
    #Convert bytes to string
    results = lines.decode()
    #Split string on whitespace
    results = results.split()

    if len(results) == 5: #didn't detected a face
        return []

    else:
        #Save the confidence percentage
        confidence = results[1]
        x = int(float(results[3]))  #left_x
        y = int(float(results[5])) #top_y
        w = int(float(results[7])) #width
        h = int(float(results[9][:-1])) #height, -1 to loose the ')' at the end
        #Crop face
        image = cv.imread(image)
        im_h, im_w = np.shape(image)[:2]

        ymin = max(0,y)
        ymax = min(h + ymin, im_h)
        xmin = max(0,x)
        xmax = min(w + xmin, im_w)

        face = image[ymin:ymax,xmin:xmax,:]

        return [face, confidence, xmin, xmax, ymin, ymax]

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def get_pose(face, pose_classifier):
    HOGs = get_HOGs([face])
    pose = pose_classifier.predict(HOGs) # 0 - frontal 1 - tilted 2 - profile 3 - tilted (-) 4 - profile (-)
    return pose[0]

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def get_lms(img_path, fitters, pose, xmin, xmax, ymin, ymax):
    menpo_image = mio.import_image(img_path)
    open_cv_image = cv.imread(img_path)

    if pose == 0:
        fitter = fitters[0]
    elif pose == 1 or pose == 3:
        fitter = fitters[1]
    elif pose == 2 or pose == 4:
        fitter = fitters[2]
    if pose == 3 or pose == 4: # negative ones
        menpo_image.mirror()

    bb = menpo.shape.bounding_box((ymin,xmin), (ymax, xmax))
    shape = fitter.fit_from_bb(menpo_image, bb).final_shape

    lms = []
    for (y, x) in shape.as_vector().reshape((-1, 2)):
        lms.append((int(x),int(y)))
        if pose == 3 or pose == 4:
            flip_lms = []
            h, w = np.shape(open_cv_image)[:2]
            for pt in lms:
                flip_lms.append((w - pt[0], pt[1]))
            lms = flip_lms


    corrected_lms = [] # Correct any lms that is out of the bounding box
    for pt in lms:
        x = max(min(pt[0], xmax), xmin)

        y = max(min(pt[1], ymax), ymin)

        corrected_lms.append((x,y))

    """
    TEST
    ==================

    copy_image = open_cv_image.copy()
    plt.figure()

    for pt in corrected_lms:
        cv.circle(copy_image, pt, 3, (255,0,0), thickness=-1, lineType=cv.LINE_AA)

    cv.imwrite(img_path.split('/')[-1], copy_image)
		"""
    return corrected_lms

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def deal_excel(excel):
	data_pain = pd.read_excel(excel)
	data = np.asarray(data_pain).T[1:]
	keys = np.asarray(data_pain.keys())[1:]
	data = np.concatenate((keys.reshape(-1,1), data), axis = 1)

	for i, label in enumerate(['ears', 'orbital', 'eyelid', 'sclera', 'noistrils', 'total']):
		ans = data[:, i+1]
		new_list = []
		for j in ans:
			if 'nan' not in j:
				#print(j)
				new_list.append(literal_eval(j))
		ans = np.vstack(new_list)
		mean = np.mean(ans, axis = 0)
		#print(np.shape(mean))
		print(label,'\n=======')
		print('GT: ', mean[0], ' PRED: ', mean[1], 'MSE: ', mean[2])


def get_pain(image, landmarks, head_pose):
    #print(head_pose)
    #Initialize a dataframe to save all the predictions to
    total_df = pd.DataFrame()

    #Select list of pain features to predict and the string format of the head pose, saved Orbital tightening with a typo, so this is correct
    if int(head_pose) == 1 or int(head_pose) == 3:
        pain_to_predict = [('Ears', [0,1], list(range(10))), ('Orbital tightning', [2], list(range(10, 16))), ('Angulated upper eyelid', [2], list(range(10, 16))), ('Sclera', [2], list(range(10, 16))), ('Nostrils', [3], list(range(16, 22))), ('Corners of the mouth', [4], list(range(22, 25)))]
        str_head_pose = 'tilted'
    elif int(head_pose) == 2 or int(head_pose) == 4:
        pain_to_predict = [('Ears', [0], list(range(6))), ('Orbital tightning', [1], list(range(6, 12))), ('Angulated upper eyelid', [1], list(range(6, 12))), ('Sclera', [1], list(range(6, 12))), ('Nostrils', [2], list(range(12, 18))), ('Corners of the mouth', [3], list(range(18, 22)))]
        str_head_pose = 'side'
    elif int(head_pose) == 0:
        pain_to_predict = [('Ears', [0,1], list(range(10))), ('Orbital tightning', [2, 4], [10, 11, 12, 13, 14, 15, 22, 23, 24, 25, 26, 27]), ('Angulated upper eyelid', [2,4], [10, 11, 12, 13, 14, 15, 22, 23, 24, 25, 26, 27]), ('Sclera', [2,4], [10, 11, 12, 13, 14, 15, 22, 23, 24, 25, 26, 27]), ('Nostrils', [3, 5], [16, 17, 18, 19, 20, 21, 28, 29, 30, 31, 32, 33])]
        str_head_pose = 'front'
    #Extract the HOG, LBP, SIFT and CNN feautures per ROI after aligning the landmarks to the corresponding mean shape
    HOG_all, SIFT_all, CNN_all = extract_features_prototype(image, landmarks, str_head_pose)
        #Create a confidence Dataframe
    # confidence_df = pd.DataFrame()
    # confidence_df['HOG'] = k_nearest_confidence_score_extraction('HOG', 'horse', input_head_pose = str_head_pose, X_test = prediction_HOG)
    #
    # confidence_df['LBP'] = k_nearest_confidence_score_extraction('LBP', 'horse', input_head_pose = str_head_pose, input = LBP)
    #
    # confidence_df['SIFT'] = k_nearest_confidence_score_extraction('SIFT', 'horse', input_head_pose = str_head_pose, X_test = prediction_SIFT)
    #
    # confidence_df['CNN'] = k_nearest_confidence_score_extraction('CNN', 'horse', input_head_pose = str_head_pose, X_test = prediction_CNN)

    #Iterate over the pain features
    pain = {}
    fuse_results_list = []

    for pain_feature, roi_features, sift_roi_features in pain_to_predict:
        #Load the pain estimator models
        HOG_SVR = pickle.load(open(os.path.join(MODELS,'SVC_%s_%s_HOG_horse.sav' % (str_head_pose, pain_feature)), 'rb'))
        # LBP_SVR = pickle.load(open(os.path.join(MODELS,'SVC_%s_%s_LBP_horse.sav' % (str_head_pose, pain_feature)), 'rb'))
        CNN_SVR = pickle.load(open(os.path.join(MODELS,'SVC_%s_%s_CNN_horse.sav' % (str_head_pose, pain_feature)), 'rb'))
        SIFT_SVR = pickle.load(open(os.path.join(MODELS,'SVC_%s_%s_SIFT_horse.sav' % (str_head_pose, pain_feature)), 'rb'))
        #Load the weights
        # weights = np.load(os.path.join(MODELS,'fusion_weights_%s_%s.npy' % (str_head_pose, pain_feature)))
        #Create an empty dataframe
        df = pd.DataFrame(columns = ['HOG', 'SIFT', 'CNN'])
        confidence = pd.DataFrame(columns = ['HOG', 'SIFT', 'CNN'])
        #Predict the pain score and save it to the dataframe
        HOG = HOG_all[roi_features].flatten()
        #print(HOG.shape)
        HOG = HOG.reshape(1, -1)
        prediction_HOG = HOG_SVR.predict(HOG)
        df['HOG'] = prediction_HOG
        confidence_HOG = HOG_SVR.predict_proba(HOG)
        #print(confidence)
        confidence['HOG'] = [confidence_HOG[0][int(prediction_HOG)]]

        total_df['HOG_' + pain_feature] = prediction_HOG
        # LBP = LBP.reshape(1, -1)
        # prediction_LBP = LBP_SVR.predict(LBP)
        # df['LBP'] = prediction_LBP
        # total_df['LBP_' + pain_feature] = prediction_LBP
        SIFT = SIFT_all[sift_roi_features].flatten()
        SIFT = SIFT.reshape(1, -1)
        #print(SIFT.shape)
        prediction_SIFT = SIFT_SVR.predict(SIFT)
        df['SIFT'] = prediction_SIFT
        confidence_sift = SIFT_SVR.predict_proba(SIFT)
        confidence['SIFT'] = [confidence_sift[0][int(prediction_SIFT)]]
        total_df['SIFT_' + pain_feature] = prediction_SIFT
        #print(confidence)
        CNN = CNN_all[roi_features].flatten()
        #print(CNN.shape)
        CNN = CNN.reshape(1, -1)
        prediction_CNN = CNN_SVR.predict(CNN)
        df['CNN'] = prediction_CNN
        confidence_cnn = CNN_SVR.predict_proba(CNN)

        confidence['CNN'] = [confidence_cnn[0][int(prediction_CNN)]]
        #print(confidence)
        total_df['CNN_' + pain_feature] = prediction_CNN

        #Fuse the results of all the extracted feature types given the weights
        FUSION = pickle.load(open(os.path.join(MODELS,'fusion_%s_%s.sav' % (str_head_pose, pain_feature)), 'rb'))
        results = FUSION.predict(df)
        #print(results)
        fuse_results_list.append(results)
        result_confidence = FUSION.predict(confidence)
        #print(result_confidence)

        #Fuse the confidences with the same weights
        pain[pain_feature] = {'score': round(float(results), 2), 'confidence': round(float(result_confidence), 2)}
    #print(total_df)
    #print(fuse_results_list)
    TOTAL = pickle.load(open(os.path.join(MODELS,'total_score_linear_regression_%s.sav' % (str_head_pose)), 'rb'))
    total_score = TOTAL.predict(total_df)
    if total_score > 4:
        pain_level = 'in pain'
    else:
        pain_level = 'not in pain'
    # total_score, pain_level = predict_total_score(str_head_pose, total_df)
    # total_score = np.mean(fuse_results_list)

    pain['Total pain'] = {'pain':pain_level, 'pain level': round(float(total_score), 2)}

    return pain

#%%============================================================================
#                               MAIN
#============================================================================

def get_mse(image_name, pain_gt, pain_auto, pain_manual, head_pose):

	image_info = [image_name]

	keywords = ['Ears', 'Orbital tightning', 'Angulated upper eyelid', 'Sclera','Nostrils']
	if head_pose != 0:
		keywords.append('Corners of the mouth')

	for keyword in keywords:
		gt = pain_gt[keyword][int(image_name) - 1]

		if gt != -1 :
			pred_manual = get_rounded([pain_manual[keyword]['score']])[0]
			mse_manual = mean_squared_error([gt], [pred_manual])
			pred_auto = get_rounded([pain_auto[keyword]['score']])[0]
			mse_auto = mean_squared_error([gt], [pred_auto])

		else:
			pred_manual = -1
			mse_manual = -1
			pred_auto = -1
			mse_auto = -1

		image_info.append([gt, pred_manual, mse_manual, pred_auto, mse_auto])
	if head_pose == 0:
		image_info.append([-1,-1, -1, -1, -1])

	gt_score = pain_gt['Total score'][int(image_name) - 1]

	pred_score_manual = pain_manual['Total pain']['pain level']
	mse_manual = mean_squared_error([gt_score], [pred_score_manual])

	pred_score_auto = pain_auto['Total pain']['pain level']
	mse_auto = mse_classifictaion_fusion([gt_score], [pred_score_auto])

	image_info.append([gt_score, pred_score_manual, mse_manual, pred_score_auto, mse_auto])
	return np.hstack(image_info)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Pain assessment')
    parser.add_argument('img', metavar='img_path', type= str, help = 'Relative path to the image(s)')
    parser.add_argument('--pose', metavar='pose', type= int, help = 'Quantitative pose\n0 - frontal\n1 - tilted\n2 - profile\n3 - tilted (-)\n4 - profile (-)')
    args = parser.parse_args()

    if args.pose:
        head_pose = args.pose
    else:
        head_pose = []

    img_path = os.path.join(os.getcwd(), args.img)
    paths = glob.glob(img_path)
    return head_pose, paths

#def main(head_pose, paths):
pose_classifier, fitters = load_models()

#%%

#for indexes, name in [[indexes_0,'0'], [indexes_30, '30'], [indexes_60, '60']]:
for indexes, name in [[indexes_30, '30'], [indexes_60, '60']]:
	pain_data = []
	pain_data.append(['img_name',
										'ears gt', 'ears pred manual', 'ears mse manual', 'ears pred auto', 'ears mse auto',
										'orbital gt', 'orbital pred manual', 'orbital mse manual', 'orbital pred auto', 'orbital mse auto',
										'eyelid gt', 'eyelid pred manual', 'eyelid mse manual', 'eyelid pred auto', 'eyelid mse auto',
										'sclera gt', 'sclera pred manual', 'sclera mse manual', 'sclera pred auto', 'sclera mse auto',
										'nostrils gt', 'nostrils pred manual', 'nostrils mse manual', 'nostrils pred auto', 'nostrils mse auto',
										'mouth gt', 'mouth pred manual', 'mouth mse manual', 'mouth pred auto', 'mouth mse auto',
										'total score gt', 'total score pred manual', 'total score mse manual', 'total score pred auto', 'total score mse auto'])


	#%%

	for index in test_indexes:
		print('INDEX : ', index)
		index = index - 1
		gt_lms = data.values[index][-1]
		gt_pose = data.values[index][2]

		if gt_pose == 0:
			head_pose = 0
		elif gt_pose == 30:
			head_pose = 1
		elif gt_pose == 60:
			head_pose = 2
		elif gt_pose == -30:
			head_pose = 3
		elif gt_pose == -60:
			head_pose = 4

		img_path = data.values[index][0]
		img_name = img_path.split('/')[-1].split('.')[0]
		img = cv.imread(img_path)
		gt_lms = np.vstack(gt_lms)
		lms_x = gt_lms[:,0]
		lms_y = gt_lms[:,1]

		error = 0.10
		img_h, img_w = img.shape[:2]
		x_min =  max(0,int(min(lms_x) - error * img_w))
		x_max = min(img_w, int(max(lms_x) + error * img_w))
		y_min = max(0, int(min(lms_y) - error * img_h))
		y_max = min(img_h, int(max(lms_y) + error * img_h))

		"""
		start = time.time()
		results =  detect_face(img_path)
		if len(results) == 0: # If it doesn't detect a face
			print('Warning: no face detected in image %s ' % image_name)
			break

		else:
			face, confidence, xmin, xmax, ymin, ymax = results
			end = time.time()
			detection_time.append(end-start)

			# Pose estimation
			start = time.time()
			if head_pose == []:
				pose_source = 'automaticaly estimated'
				head_pose = get_pose(face, pose_classifier)
			else:
				pose_source = 'manual'
				end = time.time()
				pose_time.append(end-start)
		"""
		# Lms detection
		start = time.time()
		lms = get_lms(img_path, fitters, head_pose, x_min, x_max, y_min, y_max)
		end = time.time()
		lms_time.append(end-start)

		# Pain estimation
		start = time.time()

		pain_auto = get_pain(img, np.vstack(lms), head_pose)
		end = time.time()
		pain_time.append(end-start)

		pain_manual = get_pain(img, np.vstack(gt_lms), head_pose)

		# Make the JSON FILE
		img_info = {}
		img_info['bounding_box'] = {
	                    'xmin': x_min,
	                    'ymin': y_min,
	                    'xmax': x_max,
	                    'ymax': y_max}

		#img_info['pose'] = {'source': pose_source, 'head pose': head_pose[0]}
		img_info['pose'] = {'source': '-', 'head pose': head_pose}

		dict_lms = {}
		for i,pt in enumerate(lms):
			dict_lms[i] = (pt[0],pt[1])
			img_info['lms (x,y)'] = dict_lms
			img_info['pain'] = pain_manual

		with open(os.path.join(TEST,img_name + '.txt'), 'w') as outfile:
			json.dump(img_info, outfile, indent = 4, sort_keys=True)

		#shutil.copy(img_path, os.path.join(TEST, img_name + '.jpg'))

		pain_data.append(np.hstack(get_mse(img_name, pain_gt, pain_auto, pain_manual, head_pose)))


	with open('pain_test_%s.pkl' % name, 'wb') as f:
		pickle.dump(pain_data, f)

	#print('detection time: ', np.mean(detection_time))
	#print('pose time: ', np.mean(pose_time))
	#print('lms time: ', np.mean(lms_time))
	#print('pain time: ', np.mean(pain_time))
	"""
	if __name__ == '__main__':
	    # Uncomment the following lines to run on the terminal
	    #pose, paths = parse_arguments()
	    #main(pose, paths)
	    main([], test)
	"""
