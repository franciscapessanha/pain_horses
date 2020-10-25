#%%============================================================================
#                       IMPORTS AND INITIALIZATIONS
#============================================================================

import numpy as np
import os
import glob
from skimage.feature import hog
import cv2 as cv
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pickle
from sklearn.metrics import confusion_matrix

SVM_MODEL = 'pose_classifier_0.10.sav'
MODELS = os.path.join(os.getcwd(), 'models')

DATASET = os.path.join(os.getcwd(), '..', 'dataset')

data = pickle.load(open( os.path.join(DATASET, 'lms_annotations.pkl'), "rb" ))

ANIMAL = 'horse'

if ANIMAL == 'horse':
    ABS_POSE = os.path.join(DATASET,'abs_pose')
    test_0 = glob.glob(os.path.join(ABS_POSE,'frontal','test','*.png'))
    test_30 = glob.glob(os.path.join(ABS_POSE,'tilted','test', '*.png'))
    test_60 = glob.glob(os.path.join(ABS_POSE,'profile','test','*.png'))

if ANIMAL == 'donkey':
    ABS_POSE = os.path.join(DATASET,'abs_pose_donkeys')
    test_0 = glob.glob(os.path.join(ABS_POSE,'0_30/','*.jpg'))
    test_30 = glob.glob(os.path.join(ABS_POSE,'30_60/', '*.jpg'))
    test_60 = glob.glob(os.path.join(ABS_POSE,'60_90/','*.jpg'))

test_0 = [t.split('/')[-1].replace('png','jpg') for t in test_0]
test_30 = [t.split('/')[-1].replace('png','jpg')  for t in test_30]
test_60 = [t.split('/')[-1].replace('png','jpg')  for t in test_60]

test = np.concatenate((test_0, test_30, test_60)).tolist()

test_info = []
train_info = []
for info in data.values:
    found = False
    if info[1] == 'horse':
	    for t in test:
	        if '/' + t in info[0]:
	            test_info.append(info)
	            found = True
	    if found == False:
	        train_info.append(info)



FOLDS = 3

#cells_per_blocks = [1, 2, 3, 4]
#pixels_per_cells = [8, 16]
#KERNELS = ['linear', 'rbf']

orientations = [9]
pixels_per_cells = [8]
cells_per_blocks = [4]
KERNELS = ['linear']

#MODE in ['cross_val', 'final_model']

MODE = 'final_model'

EXAMPLES = os.path.join(os.getcwd(), 'bad_pose_examples')

if os.path.exists(EXAMPLES) is not True:
     os.mkdir(EXAMPLES)

def crop_image(img, lms, pose):

	error = 0.10
	lms = np.vstack(lms)
	lms_x = lms[:,0]
	lms_y = lms[:,1]

	img_h, img_w = img.shape[:2]
	x_min =  max(0,int(min(lms_x) - error * img_w))
	x_max = min(img_w, int(max(lms_x) + error * img_w))

	y_min = max(0, int(min(lms_y) - error * img_h))
	y_max = min(img_h, int(max(lms_y) + error * img_h))

	img_crop = img[y_min : y_max, x_min : x_max]
	crop_h, crop_w = img_crop.shape[:2]

	if crop_h >  crop_w:
		new_h = 855
		new_w = int((crop_w * new_h) / crop_h)

	elif crop_w >= crop_h:
		new_w = 855
		new_h = int((crop_h * new_w) / crop_w)


	img_resize = cv.resize(img_crop, (new_w, new_h))
	lms_resize = []
	for pt in lms:
		new_pt = ((pt[0] - x_min) * new_w/crop_w, (pt[1] - y_min) * new_h/crop_h)
		#new_pt = ((pt[0] - x_min), (pt[1] - y_min))
		lms_resize.append(new_pt)


	lms_resize = np.vstack(lms_resize)


	return img_resize, lms_resize

#%%============================================================================
#                                LOAD DATA
#==============================================================================

images_0 = []
images_30 = []
images_60 = []
images_m30 = [] # negative pose
images_m60 = [] # negative pose

test_images_0 = []
test_images_30 = []
test_images_60 = []
test_images_m30 = [] # negative pose
test_images_m60 = [] # negative pose

def save_resized_img_and_pts():
    for img_info in train_info:
        #filter the images that don't have landmark annotations!
        if img_info[-2] is not None and img_info[1] == ANIMAL:
            pose = img_info[2]
            img_name = img_info[0].split('/')[-1]
            img = cv.imread(os.path.join(os.getcwd(), '..', img_info[0]))
            lms = img_info[-1]
            #print(img_info[0])
            img, lms = crop_image(img, lms, pose)

            if pose == 0:
               images_0.append(img)
            elif pose == 30:
                images_30.append(img)
            elif pose == 60:
                images_60.append(img)
            elif pose == -60:
                images_m60.append(img)
            elif pose == -30:
                images_m30.append(img)

def save_resized_img_and_pts_test():
    for img_info in test_info:
        #filter the images that don't have landmark annotations!
        if img_info[-2] is not None and img_info[1] == ANIMAL:
            pose = img_info[2]
            img_name = img_info[0].split('/')[-1]
            img = cv.imread(os.path.join(os.getcwd(), '..', img_info[0]))
            lms = img_info[-1]
            #print(img_info[0])
            img, lms = crop_image(img, lms, pose)


            if pose == 0:
               test_images_0.append(img)
            elif pose == 30:
                test_images_30.append(img)
            elif pose == 60:
                test_images_60.append(img)
            elif pose == -60:
                test_images_m60.append(img)
            elif pose == -30:
                test_images_m30.append(img)


#%%============================================================================
#                               AUXILIAR FUNCTIONS
#==============================================================================

def get_val_train(images, fold, FOLDS):
    images_per_fold= len(images) // FOLDS
    val_start = fold * images_per_fold
    val_end = (fold + 1) * images_per_fold
    val = images[val_start:val_end]
    train = images[:val_start] + images[val_end:]

    return train, val

def get_HOGs(imgs, imgs_to_flip = []):
    o = 9
    ppc = 8
    cpb = 4
    all_HOG = []

    for img in imgs:
        img = cv.resize(img, (128,128))
        all_HOG.append(hog(img,  orientations= o, pixels_per_cell=(ppc,ppc), cells_per_block=(cpb, cpb), multichannel=True))

    if len(imgs_to_flip) != 0:
        for img_flip in imgs_to_flip:
            img = cv.flip(img_flip, 1)
            img = cv.resize(img, (128,128))
            all_HOG.append(hog(img,  orientations= o, pixels_per_cell=(ppc,ppc), cells_per_block=(cpb, cpb), multichannel=True))

    return all_HOG

#%%============================================================================
#                              MAIN
#==============================================================================

def main():

    save_resized_img_and_pts()
    save_resized_img_and_pts_test()

    for o in orientations:
        for ppc in pixels_per_cells:
            for cpb in cells_per_blocks:
                for kernel in KERNELS:
                    images_0_HOGs = get_HOGs(images_0)
                    images_30_HOGs = get_HOGs(images_30, images_m30)
                    images_60_HOGs = get_HOGs(images_60, images_m60)
                    images_m30_HOGs = get_HOGs(images_m30, images_30)
                    images_m60_HOGs = get_HOGs(images_m60, images_60)

                    if MODE == 'cross_val':
                        for fold in range(FOLDS):
                            #print('orientation: ', o)
                            #print('pixels per cell: ', ppc)
                            #print('cells per block: ', cpb)
                            #print('kernel: ', kernel)
                            #print('fold: ', fold)
                            train_0, val_0 =  get_val_train(images_0_HOGs, fold, FOLDS)
                            train_30, val_30 =  get_val_train(images_30_HOGs, fold, FOLDS)
                            train_60, val_60 =  get_val_train(images_60_HOGs, fold, FOLDS)
                            train_m30, val_m30 =  get_val_train(images_m30_HOGs, fold, FOLDS)
                            train_m60, val_m60 =  get_val_train(images_m60_HOGs, fold, FOLDS)


                            x_val = np.concatenate((val_0, val_30, val_60, val_m30, val_m60), axis = 0)
                            y_val = np.concatenate((np.zeros((len(val_0), 1)),np.ones((len(val_30), 1)), 2 * np.ones((len(val_60), 1)), 3 * np.ones((len(val_m30), 1)), 4 * np.ones((len(val_m60), 1))), axis = 0).ravel()

                            x_train = np.concatenate((train_0, train_30, train_60, train_m30, train_m60), axis = 0)
                            y_train = np.concatenate((np.zeros((len(train_0), 1)),np.ones((len(train_30), 1)), 2 * np.ones((len(train_60), 1)), 3 * np.ones((len(train_m30), 1)), 4 * np.ones((len(train_m60), 1))), axis = 0).ravel()

                            val_HOG = np.vstack(x_val)
                            train_HOG = np.vstack(x_train)


                            modelSVM = SVC(kernel = kernel, gamma = 'auto', decision_function_shape= 'ovo',class_weight='balanced')
                            modelSVM.fit(train_HOG, y_train)
                            y_pred = modelSVM.predict(val_HOG)

                            target_names = ['frontal', 'tilted', 'profile', 'tilted_minus', 'profile_minus']
                            print(classification_report(y_val, y_pred, target_names=target_names))

                    elif MODE == 'final_model':
        #%%
                            filename = SVM_MODEL
                            file_path = os.path.join(MODELS, filename)

                            if os.path.exists(os.path.join(MODELS, filename)) is not True:
                                x_train = np.concatenate((images_0_HOGs, images_30_HOGs, images_60_HOGs, images_m30_HOGs, images_m60_HOGs), axis = 0)
                                y_train = np.concatenate((np.zeros((len(images_0_HOGs), 1)),np.ones((len(images_30_HOGs), 1)), 2 * np.ones((len(images_60_HOGs), 1)), 3 * np.ones((len(images_m30_HOGs), 1)), 4 * np.ones((len(images_m60_HOGs), 1))), axis = 0).ravel()
                                x_train = np.vstack(x_train)
                                modelSVM = SVC(kernel = kernel, gamma = 'auto', decision_function_shape= 'ovo',class_weight='balanced')
                                modelSVM.fit(x_train, y_train)

                                pickle.dump(modelSVM, open(file_path, 'wb'))

                            else:
                                modelSVM = pickle.load(open(file_path, 'rb'))
    #%%
                            test_0_HOGs = get_HOGs(test_images_0)
                            test_30_HOGs = get_HOGs(test_images_30)
                            test_60_HOGs = get_HOGs(test_images_60)
                            test_m30_HOGs = get_HOGs(test_images_m30)
                            test_m60_HOGs = get_HOGs(test_images_m60)


                            #test_images = np.concatenate((test_images_0,test_images_30, test_images_60, test_images_m30, test_images_m60), axis = 0)
                            x_test = np.concatenate((test_0_HOGs, test_30_HOGs, test_60_HOGs, test_m30_HOGs, test_m60_HOGs), axis = 0)
                            y_test = np.concatenate((np.zeros((len(test_0_HOGs), 1)),np.ones((len(test_30_HOGs), 1)), 2 * np.ones((len(test_60_HOGs), 1)),  3 * np.ones((len(test_m30_HOGs), 1)),  4 * np.ones((len(test_m60_HOGs), 1))), axis = 0).ravel()
                            y_pred = modelSVM.predict(x_test)

                            target_names = ['frontal', 'tilted', 'profile', 'tilted_minus', 'profile_minus']
                            print(classification_report(y_test, y_pred, target_names=target_names))

                            print(confusion_matrix(y_test, y_pred))
                            #%%
                            test_images = []
                            for list_ in [test_images_0, test_images_30, test_images_60, test_images_m30, test_images_m60]:
                                for element in list_:
                                    test_images.append(element)

                            for i in range(len(y_pred)):
                                if y_test[i] != y_pred[i]:
                                    filename = 'pred_%d_vs_label_%d-%d.jpg' %(y_pred[i], y_test[i], i)
                                    cv.imwrite(os.path.join(EXAMPLES, filename), test_images[i])


main()
