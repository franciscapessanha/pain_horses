
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
from scipy.spatial import distance
#from data_augmentation.utils import *
import random
import pandas as pd

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

MODE = 'final_model'
if not os.path.exists(ABS_POSE):
    os.mkdir(ABS_POSE)

for folder in ['frontal', 'tilted','profile']:
    path = os.path.join(ABS_POSE,folder)
    if not os.path.exists(path):
        os.mkdir(path)

    for k in range(N_FOLDS):
        for AUG in  [0.3, 0.5, 0.7, 0.9]:
            sub_path = os.path.join(path,'v2_data_%.1f_%d' % (AUG, k))
            if not os.path.exists(sub_path):
                os.mkdir(sub_path)
    for AUG in  [0.5]:
        sub_path = os.path.join(path,'v2_data_%.1f' % (AUG))
        if not os.path.exists(sub_path):
            os.mkdir(sub_path)

TEMP = os.path.join(os.getcwd(), 'temp')

if not os.path.exists(TEMP):
    os.mkdir(TEMP)
#%%============================================================================
                               #FUNCTIONS
#==============================================================================
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

def save_dataframe(all_info, name):
    df_ = pd.DataFrame(columns=columns)
    for info in all_info:
        img_path = info[0]
        #print(img_path)
        lms = np.vstack(info[1])
        try:
            img = cv.imread(img_path)
            lms_x = lms[:,0]
            lms_y = lms[:,1]

            img_h, img_w = img.shape[:2]
            x_min =  max(0,int(min(lms_x)))
            x_max = min(img_w, int(max(lms_x)))

            y_min = max(0, int(min(lms_y)))
            y_max = min(img_h, int(max(lms_y)))

            roll = float(info[2])
            pitch = float(info[3])
            yaw = float(info[4])

            dic = {'path': img_path,
                     'bbox_x_min': x_min,
                     'bbox_y_min': y_min,
                     'bbox_x_max': x_max,
                     'bbox_y_max': y_max,
                     'yaw': yaw,
                     'pitch': pitch,
                     'roll': roll}
            df_.loc[len(df_)] = dic
        except:
            pass
        df_.to_csv (name + '.csv', index = False, header=True)

def get_list_angles(angles_train, all_yaw, all_pitch, all_roll):
    yaw_train = angles_train[:,-1].astype(np.float)
    counts_yaw, bins_yaw, _ = plt.hist(yaw_train, bins=36, range=(-95,95))
    max_counts = max(counts_yaw)
    n_yaw = 0

    for i in range(len(bins_yaw) - 1):
        if counts_yaw[i] != 0:
            aug_factor = (max_counts/counts_yaw[i])**alpha
            n_new_values = int(counts_yaw[i] * (aug_factor - 1))
        elif counts_yaw[i] == 0:
            n_new_values = 0
            #print('entrou')
        n_yaw += n_new_values
        for j in range(n_new_values):
            all_yaw.append(random.uniform(bins_yaw[i], bins_yaw[i + 1]))

    pitch_train = angles_train[:,-2].astype(np.float)
    counts_pitch, bins_pitch, _ = plt.hist(pitch_train, bins=36, range=(-95,95))
    """
    for i in range(len(bins_pitch) - 1):
        if counts_pitch[i] != 0:
            n_new_values = math.ceil(((counts_pitch[i]*n_yaw)/sum(counts_pitch)))
        elif counts_pitch[i] == 0:
            n_new_values = 0
        for j in range(n_new_values):
            all_pitch.append(random.uniform(bins_pitch[i], bins_pitch[i + 1]))

    roll_train = angles_train[:,-3].astype(np.float)
    counts_roll, bins_roll, _ = plt.hist(roll_train, bins=36, range=(-95,95))

    for i in range(len(bins_roll) - 1):
        if counts_roll[i] != 0:
            aug_factor = len(all_yaw)/sum(counts_roll)
            n_new_values = math.ceil(((counts_roll[i]*n_yaw)/sum(counts_roll)))
        elif counts_roll[i] == 0:
            n_new_values = 0

        for j in range(n_new_values):
            all_roll.append(random.uniform(bins_roll[i], bins_roll[i + 1]))
    """
    max_counts = max(counts_pitch)
    all_new_values = []
    for i in range(len(bins_pitch) - 1):
        if counts_pitch[i] != 0:
            aug_factor = (max_counts/counts_pitch[i])**alpha
            all_new_values.append(math.ceil(counts_pitch[i] * (aug_factor - 1)))
        elif counts_pitch[i] == 0:
            all_new_values.append(0)
    for i in range(len(bins_pitch) - 1):
        a = math.ceil((all_new_values[i] * n_yaw)/sum(all_new_values))
        for j in range(a):
            all_pitch.append(random.uniform(bins_pitch[i], bins_pitch[i + 1]))


    roll_train = angles_train[:,-3].astype(np.float)
    counts_roll, bins_roll, _ = plt.hist(roll_train, bins=18, range=(-95,95))
    max_counts = max(counts_roll)
    all_new_values = []
    for i in range(len(bins_roll) - 1):
        if counts_roll[i] != 0:
            aug_factor = (max_counts/counts_roll[i])**alpha
            all_new_values.append(math.ceil(counts_roll[i] * (aug_factor - 1)))
        elif counts_roll[i] == 0:
            all_new_values.append(0)
    n_roll = 0
    for i in range(len(bins_roll) - 1):
        a = math.ceil((all_new_values[i] * n_yaw)/sum(all_new_values))
        for j in range(a):
            n_roll +=1
            all_roll.append(random.uniform(bins_roll[i], bins_roll[i + 1]))

    return all_yaw, all_pitch, all_roll

def generate_img(all_yaw, all_roll, all_pitch, folder, i, pose):
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

    path_bg = random.choice(list(BACKGROUND))
    img_background = cv.imread(path_bg)
    #print(np.shape(img_background))

    R_y = abs(all_yaw[i])
    R_x = all_roll[i]
    R_z = all_pitch[i]

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
    #silhouete_lms = []
    #for pt in silhouete:
    #    silhouete_lms.append(find_closest_point(lms, pt))

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

    for pt in lms_resize:
        cv.circle(img_with_pts,tuple((int(pt[0]), int(pt[1]))), 3, (255,0,0), -1)

    cv.imwrite(os.path.join(EX_FOLDER, pose +'_' + str(k) + '_' + str(i) + '.png'), img_with_pts)

    cv.imwrite(os.path.join(folder, str(i) + '.png'), new_img)
    create_pts(lms_resize, os.path.join(folder, str(i) + '.pts'))

    return [os.path.join(folder, str(i) + '.png'), lms_resize, R_x, R_z, R_y]


#%%============================================================================
                               #MAIN
#==============================================================================
for alpha in [0.3]:

    COLORS =  os.path.join(DATASET, '3D_annotations', 'colors')
    SHAPES =  os.path.join(DATASET, '3D_annotations', 'shapes')
    ANGLES =  os.path.join(DATASET, '3D_annotations', 'angles')
    MODELS =  os.path.join(DATASET, '3D_annotations', 'models')


    data = pickle.load(open( os.path.join(DATASET, 'lms_annotations.pkl'), "rb" ))
    train_set = glob.glob((os.path.join(DATASET, 'cross_val', 'train', '*.jpg')))

    columns = ['path', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'yaw', 'pitch', 'roll']

    BACKGROUND = glob.glob(os.path.join(DATASET, 'flickr_backgrounds', '*.png'))

    angles = glob.glob(os.path.join(ANGLES, '*.pickle'))
    shapes = glob.glob(os.path.join(SHAPES, '*.obj'))
    colors_list = glob.glob(os.path.join(COLORS, '*.pickle'))

    if MODE == 'cross_val':
        print('AUG: ', alpha)
        for k in range(N_FOLDS):
            print('k: ', k)
            for label in ['frontal', 'tilted', 'profile']:
            #for label in ['profile']:
                print('label: ', label)
                all_train = []
                all_val = []

                all_yaw = []
                all_roll = []
                all_pitch = []
                all_info_train = []
                angles_complete_train = np.vstack(np.load(open(os.path.join(ANGLES, '%s_pain_train_angles.pickle' % (label)), 'rb'), allow_pickle = True))
                angles_val = np.vstack(np.load(open(os.path.join(ANGLES, '%s_pain_val_fold_%d_angles.pickle' % (label, k)), 'rb'), allow_pickle = True))
                angles_train = np.vstack([i for i in angles_complete_train if i not in angles_val])

                all_yaw, all_pitch, all_roll = get_list_angles(angles_train, all_yaw, all_pitch, all_roll)
                random.shuffle(all_yaw)
                random.shuffle(all_pitch)
                random.shuffle(all_roll)

                list_ = glob.glob(os.path.join(os.path.join(ABS_POSE, label, 'v2_data_%.1f_%d' % (alpha, k) ,  '*.pts')))

                for i in range(len(all_yaw)):
                    prefix = 'v2_data_%.1f_%d' % (alpha, k)
                    folder = os.path.join(ABS_POSE, label, 'v2_data_%.1f_%d' % (alpha, k))
                    if os.path.join(folder,  '%d.pts' % i) not in list_:
                        print('%d / %d' %(i + 1, (len(all_yaw))))
                        all_info_train.append(generate_img(all_yaw, all_roll, all_pitch, folder, i, label))
                save_dataframe(all_info_train, prefix)


    # FINAL MODEL
    elif MODE == 'final_model':
        all_info_train = []
        all_yaw = []
        all_roll = []
        all_pitch = []
        for label in ['frontal', 'tilted', 'profile']:
            print('label: ', label)
            angles_train = np.vstack(np.load(open(os.path.join(ANGLES, '%s_pain_train_angles.pickle' % (label)), 'rb'), allow_pickle = True))
            all_yaw, all_pitch, all_roll = get_list_angles(angles_train, all_yaw, all_pitch, all_roll)

            random.shuffle(all_yaw)
            random.shuffle(all_pitch)
            random.shuffle(all_roll)

            prefix = 'v2_data_%.1f' % (alpha)

            list_ = glob.glob(os.path.join(os.path.join(ABS_POSE, label, prefix,  '*.pts')))

            for i in range(len(all_yaw)):
                if os.path.join(ABS_POSE, label, prefix,  '%d.pts' % i) not in list_:
                    folder = os.path.join(ABS_POSE, label, 'v2_data_%.1f' % (alpha))
                    print('%d / %d' %(i + 1, (len(all_yaw))))
                    all_info_train.append(generate_img(all_yaw, all_roll, all_pitch, folder, i, label))
            save_dataframe(all_info_train, prefix)