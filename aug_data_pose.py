
import os
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import pickle
import cv2 as cv
import trimesh
import glob
from data_augmentation.load_obj import *
from data_augmentation.transformations import *
from data_augmentation.utils import crop_image
import math
from utils import *
import random
import pandas as pd
import pyrender

DATASET = os.path.join(os.getcwd(), 'dataset')
ANGLES =  os.path.join(DATASET, '3D_annotations', 'angles')
N_FOLDS = 3


MODE = 'cross_val'
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

def find_closest_point(silhouete, points):
    new_points = []
    for pt in points:
        dists = []
        for s in silhouete:
            dists.append(np.sqrt((s[0] - pt[0])**2 + (s[1] - pt[1])**2))
        index = np.argmin(dists, axis=0)
        if type(index) == np.ndarray:
            index = index[0]
        new_points.append(silhouete[index])

    new_points = np.vstack(new_points)
    return new_points

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
    counts_pitch, bins_pitch, _ = plt.hist(pitch_train, bins=18, range=(-95,95))

    for i in range(len(bins_pitch) - 1):
        if counts_pitch[i] != 0:
            aug_factor = len(all_yaw)/sum(counts_pitch)
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


    """
    return all_yaw, all_pitch, all_roll

def generate_img(all_yaw, all_roll, all_pitch, name, i):
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

    R_y = all_yaw[i]
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

    mesh = trimesh.Trimesh(vertices = vertices_rot, faces=faces, process=False)
    #mesh.faces = np.fliplr(mesh.faces)
    for f in range(len(mesh.faces)):
         mesh.visual.face_colors[f] = np.asarray(colors[f])#trimesh.visual.random_color()


    scene = mesh.scene()
    scene.camera.K = np.load(shape.replace('obj', 'pickle'), 'rb', allow_pickle = True)
    scene.camera.resolution = tuple([1600, 800])
    scene.camera_transform = scene.camera.look_at(vertices_rot)

    name = 'yaw_%d_pitch_%d_roll_%d.png' % (R_y,R_z, R_x)

    png = scene.save_image(resolution = scene.camera.resolution, background = [0,0,0], visible = True)
    with open(os.path.join(TEMP, name) , 'wb') as f:
        f.write(png)
        f.close()

    if abs(R_y) < 25:
        pose = 'frontal'
    elif abs(R_y) > 50:
        pose = 'profile'
    else:
        pose = 'tilted'


    if R_y < 0:
        extension = '_indexes.pickle'
    else:
        extension = '_sim_indexes.pickle'
    img_load = cv.imread(os.path.join(TEMP, name))
    lms, _ = project_pts(mesh.vertices, scene.camera.K, external_parameters = np.dot(np.linalg.inv(RT_4x4),np.linalg.inv(scene.camera_transform))[:3,:])

    background_h, background_w = np.shape(img_background)[:2]
    img_resize, lms_resize = crop_image(img_load, lms)
    int_lms = np.vstack([lms_resize[i] for i in pickle.load(open(os.path.join(MODELS, pose + extension), 'rb'))])

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

    new_img = img_resize.copy()
    for i_h in range(head_h - 1):
        for i_w in range(head_w - 1):
            px = img_resize[i_h, i_w, :]
            if (px == np.asarray([0,0,0])).all(): # 0,0,200 and not [200,200,0] because cv2 works with BGR and not RGB
                #print('enter')
                new_img[i_h, i_w, :] = resize_background[i_h, i_w, :]

    img_path = os.path.join(POSE, prefix, str(i) + '.png')
    cv.imwrite(img_path, new_img)
    np.save(img_path.replace('.png', ''), [img_path, lms_resize, R_x, R_z, R_y])
    return [img_path, lms_resize, R_x, R_z, R_y]


#%%============================================================================
                               #MAIN
#==============================================================================
for alpha in [0.3]:

    COLORS =  os.path.join(DATASET, '3D_annotations', 'colors')
    SHAPES =  os.path.join(DATASET, '3D_annotations', 'shapes')
    ANGLES =  os.path.join(DATASET, '3D_annotations', 'angles')
    MODELS =  os.path.join(DATASET, '3D_annotations', 'models')


    POSE = os.path.join(DATASET,'pose')
    N_FOLDS = 3

    if not os.path.exists(POSE):
        os.mkdir(POSE)

    if MODE == 'cross_val':
        for k in range(N_FOLDS):
            sub_path = os.path.join(POSE,'aug_data_alpha_%.1f_%d' % (alpha, k))
            if not os.path.exists(sub_path):
                os.mkdir(sub_path)
    else:
        sub_path = os.path.join(POSE,'aug_data_alpha_%.1f' % (alpha))
        if not os.path.exists(sub_path):
            os.mkdir(sub_path)


    TEMP = os.path.join(os.getcwd(), 'temp')

    if not os.path.exists(TEMP):
        os.mkdir(TEMP)



    data = pickle.load(open( os.path.join(DATASET, 'lms_annotations.pkl'), "rb" ))
    train_set = glob.glob((os.path.join(DATASET, 'cross_val', 'train', '*.jpg')))

    columns = ['path', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'yaw', 'pitch', 'roll']

    BACKGROUND = glob.glob(os.path.join(DATASET, 'flickr_backgrounds', '*.png'))

    angles = glob.glob(os.path.join(ANGLES, '*.pickle'))
    shapes = glob.glob(os.path.join(SHAPES, '*.obj'))
    colors_list = glob.glob(os.path.join(COLORS, '*.pickle'))

    if MODE == 'cross_val':
        for k in range(N_FOLDS):
            all_train = []
            all_val = []

            all_yaw = []
            all_roll = []
            all_pitch = []
            all_info_train = []
            for label in ['frontal', 'tilted', 'profile']:
                angles_complete_train = np.vstack(np.load(open(os.path.join(ANGLES, '%s_pain_train_angles.pickle' % (label)), 'rb'), allow_pickle = True))
                angles_val = np.vstack(np.load(open(os.path.join(ANGLES, '%s_pain_val_fold_%d_angles.pickle' % (label, k)), 'rb'), allow_pickle = True))
                angles_train = np.vstack([i for i in angles_complete_train if i not in angles_val])

                all_yaw, all_pitch, all_roll = get_list_angles(angles_train, all_yaw, all_pitch, all_roll)
            random.shuffle(all_yaw)
            random.shuffle(all_pitch)
            random.shuffle(all_roll)

            list_ = glob.glob(os.path.join(os.path.join(POSE,'aug_data_alpha_%.1f_%d' % (alpha, k) ,  '*.npy')))

            for i in range(len(all_yaw)):
                prefix = 'aug_data_alpha_%.1f_%d' % (alpha, k)
                if os.path.join(POSE, prefix ,  '%d.npy' % i) not in list_:
                    print('%d / %d' %(i + 1, (len(all_yaw))))
                    all_info_train.append(generate_img(all_yaw, all_roll, all_pitch, prefix, i))
            save_dataframe(all_info_train, prefix)


    # FINAL MODEL
    elif MODE == 'final_model':
        all_info_train = []
        all_yaw = []
        all_roll = []
        all_pitch = []
        for label in ['frontal', 'tilted', 'profile']:
            angles_train = np.vstack(np.load(open(os.path.join(ANGLES, '%s_pain_train_angles.pickle' % (label)), 'rb'), allow_pickle = True))
            all_yaw, all_pitch, all_roll = get_list_angles(angles_train, all_yaw, all_pitch, all_roll)

        random.shuffle(all_yaw)
        random.shuffle(all_pitch)
        random.shuffle(all_roll)

        prefix = 'aug_data_alpha_%.1f' % (alpha)

        list_ = glob.glob(os.path.join(os.path.join(POSE, prefix,  '*.npy')))

        for i in range(len(all_yaw)):
            if os.path.join(POSE, prefix,  '%d.npy' % i) not in list_:
                print('%d / %d' %(i + 1, (len(all_yaw))))
                all_info_train.append(generate_img(all_yaw, all_roll, all_pitch, prefix, i))
        save_dataframe(all_info_train, prefix)