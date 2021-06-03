
# ============================================================================
#                          IMPORTS AND INITIALIZATIONS
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
from data_augmentation.utils import *
import math
import imutils
import numpy as np
import os
import glob
from skimage.feature import hog
import cv2 as cv
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_recall_fscore_support
from pain_estimation.get_features import *

PLOTS = os.path.join(os.getcwd(), 'plots')
if not os.path.exists(PLOTS):
    os.mkdir(PLOTS)


DATASET = os.path.join(os.getcwd(), 'dataset')
ANGLES =  os.path.join(DATASET, '3D_annotations', 'angles')

CROSS_VAL = os.path.join(os.getcwd(),'dataset','cross_val')
MODELS = os.path.join(os.getcwd(), 'pain_estimation','models')
N_FOLDS = 3

BIG_SIDE = 100

EX_FOLDER = os.path.join(os.getcwd(), 'pain_estimation', 'examples')

MODE = 'final_model' # final_model or cross_val
if not os.path.exists(MODELS):
    os.mkdir(MODELS)

for folder in ['frontal', 'tilted','profile']:
    path = os.path.join(CROSS_VAL,folder)
    if not os.path.exists(path):
        os.mkdir(path)

    for sub_folder in ['train', 'test']:
        sub_path = os.path.join(path,sub_folder)
        if not os.path.exists(sub_path):
            os.mkdir(sub_path)

data = pickle.load(open(os.path.join(DATASET, 'lms_annotations.pkl'), "rb"))

pain_scores = pd.read_excel(os.path.join(DATASET, 'pain_annotations.xlsx'), index_col=0, engine='openpyxl')

# ==============================================================================
def flip_image(img, lms, pose):
    if pose < 0:
        img = cv.flip(img, 1)
        h, w = np.shape(img)[:2]
        mirror_lms = []
        for pt in lms:
            new_pt = [w - pt[0], pt[1]]
            mirror_lms.append(new_pt)

        lms = np.vstack(mirror_lms)

    return img,lms

def flat_list(list_of_lists):
    flat_list = [item for sublist in list_of_lists for item in sublist if sublist != []]
    return flat_list

def get_x_y(set_):
    ears_x = []
    ears_rot = []
    ears_angles = []
    ears_y = []

    orbital_x = []
    orbital_rot = []
    orbital_angles = []
    orbital_y = []

    eyelid_x = []
    eyelid_rot = []
    eyelid_angles = []
    eyelid_y = []

    sclera_x = []
    sclera_rot = []
    sclera_angles = []
    sclera_y = []

    nostrils_x = []
    nostrils_rot = []
    nostrils_angles = []
    nostrils_y = []

    mouth_x = []
    mouth_rot = []
    mouth_angles = []
    mouth_y = []

    for j, angles in enumerate(set_):
        #print(angles)
        #print(j + 1, '/', len(set_))
        index = int(angles[0].split('/')[-1].split('.')[0])
        info = data.values[index - 1]
        #print(os.path.join(os.getcwd(), info[0]))
        img = cv.imread(angles[0])
        #print('info 0: ', angles[0])
        #print('shape image: ', np.shape(img))
        lms = info[-1]
        pose = info[2]

        pain_info = pain_scores.values[index][1:7]

        img,lms =  flip_image(img, lms, pose)
        lms = update_landmarks(lms, pose)

        #cv.imwrite(os.path.join(EX_FOLDER, str(random.randint(0,10000)) +  '.jpg'), img)
        #ears, eyes, eyes, eyes, mouth, nostrils
        eye, eye_rot = get_eyes(img, lms, pose)
        for i,pain in enumerate(pain_info):
                if i == 0:
                    if pain == -1:
                        ears_x.append([])
                        ears_rot.append([])
                        ears_y.append([])
                        ears_angles.append([])
                    else:
                        ears, rot = get_ears(img, lms, pose)
                        ears_x.append(ears)
                        ears_rot.append(rot)
                        y = []
                        a = []
                        #for j in range(len(ears_x[-1])):
                        for j in range(1):
                            y.append(pain)
                            a.append(angles[1:].astype(np.float))
                        ears_y.append(y)
                        ears_angles.append(a)

                elif i == 1:
                    if pain == -1:
                        orbital_x.append([])
                        orbital_rot.append([])
                        orbital_y.append([])
                        orbital_angles.append([])
                    else:
                        orbital_x.append(eye)
                        orbital_rot.append(eye_rot)
                        y = []
                        a = []
                        for j in range(len(orbital_x[-1])):
                            y.append(pain)
                            a.append(angles[1:].astype(np.float))
                        orbital_y.append(y)
                        orbital_angles.append(a)

                elif i == 2:
                    if pain == -1:
                        eyelid_x.append([])
                        eyelid_rot.append([])
                        eyelid_y.append([])
                        eyelid_angles.append([])
                    else:
                        eyelid_x.append(eye)
                        eyelid_rot.append(eye_rot)
                        y = []
                        a = []
                        for j in range(len(eyelid_x[-1])):
                            y.append(pain)
                            a.append(angles[1:].astype(np.float))
                        eyelid_y.append(y)
                        eyelid_angles.append(a)

                elif i == 3:
                        if pain == -1:
                            sclera_x.append([])
                            sclera_rot.append([])
                            sclera_y.append([])
                            sclera_angles.append([])
                        else:
                            sclera_x.append(eye)
                            sclera_rot.append(eye_rot)
                            y = []
                            a = []
                            #for j in range(len(sclera_x[-1])):
                            for j in range(1):
                                y.append(pain)
                                a.append(angles[1:].astype(np.float))
                            sclera_y.append(y)
                            sclera_angles.append(a)

                elif i == 4:
                    if pain == -1 or pose == 0:
                         mouth_x.append([])
                         mouth_rot.append([])
                         mouth_y.append([])
                         mouth_angles.append([])

                    else:
                        mouth, rot = get_mouth(img, lms, pose)
                        mouth_x.append([mouth])
                        mouth_rot.append([rot])
                        mouth_angles.append([angles[1:].astype(np.float)])
                        mouth_y.append(pain)

                elif i == 5:
                    if pain == -1:
                        nostrils_x.append([])
                        nostrils_rot.append([])
                        nostrils_y.append([])
                        nostrils_angles.append([])

                    else:
                        nostrils, rot = get_nostrils(img, lms, pose)
                        nostrils_x.append(nostrils)
                        nostrils_rot.append(rot)
                        y = []
                        a = []
                        #for j in range(len(nostrils_x[-1])):
                        for j in range(1):
                            y.append(pain)
                            a.append(angles[1:].astype(np.float))
                            #print(angles[1:])
                        nostrils_y.append(y)
                        nostrils_angles.append(a)


    return [ears_x, ears_y, ears_angles, ears_rot],[orbital_x, orbital_y, orbital_angles, orbital_rot],[eyelid_x, eyelid_y, eyelid_angles, eyelid_rot], [sclera_x, sclera_y, sclera_angles, sclera_rot], [nostrils_x, nostrils_y, nostrils_angles, nostrils_rot], [mouth_x, mouth_y, mouth_angles, mouth_rot]
