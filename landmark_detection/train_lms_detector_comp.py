"""
Created on Mon Jan 20 17:20:05 2020
@author: franciscapessanha
"""

#%%============================================================================
#                       IMPORTS AND INITIALIZATIONS
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

DATASET = os.path.join(os.getcwd(), 'dataset')
ABS_POSE = os.path.join(DATASET,'abs_pose_complete')

MODE = 'final_model'
n_pert = 70
#%%============================================================================
#                   LOAD FUNCTIONS AND CROSS-VALIDATION
#%============================================================================

def sort_files(l):
    # sort files based on the image name (numerical order)
    def f(path):
        path = path.split('/')[-1]
        path = path.split('.')[0]
        return int(path)

    return sorted(l, key=f)

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


def cross_validation_sets(lazy_list, file_list, k=5, verbose=False):
    n_images = len(file_list)
    n_images_per_fold = n_images // k
    print('Number of raw images per fold: ', n_images_per_fold)
    out_data = []

    for fold in range(k):
        val_start = fold * n_images_per_fold
        val_end = (fold + 1) * n_images_per_fold
        val_data = lazy_list[val_start:val_end]

        train_data = lazy_list[:val_start] + lazy_list[val_start + n_images_per_fold:]
        out_data.append((train_data, val_data))

        if verbose:
            print('Fold', fold, len(val_data), 'val and', len(train_data), 'train')
    return out_data

def per_lm_errors(pred, gt):
    assert(gt.n_points == gt.n_points)
    n_points = gt.n_points
    pred = pred.as_vector().copy().reshape((n_points, 2))
    gt = gt.as_vector().copy().reshape((n_points, 2))
    norm = menpofit.error.bb_avg_edge_length(gt)
    errors = []
    for i in range(n_points):
        e = np.sqrt((pred[i][0] - gt[i][0]) ** 2 + (pred[i][1] - gt[i][1]) ** 2) / norm
        errors.append(e)
    return errors

def test_eval(fitter, images, files, mean = False, pose ='',  verbose=True, save_images = False, folder = DATASET, gt = DATASET):

    error = 0
    errors = []
    plm_errors = []

    for i, image in enumerate(images):


        #print(files[i])
        lms_gt = image.landmarks['PTS'].lms.as_vector().reshape((-1, 2))
        if verbose:
            #print('Tested', i+1, ' of ', len(images), '\n')
            pass
        # use the bounding box (i.e. mean shape) to initialise
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


        if pose == '0': lms =  [*range(34)]
        if pose == '60': lms = [*range(8), *range(12,29)]
        if pose == '30': lms = [*range(23), *range(29,33)]

        open_cv_frame = image.as_PILImage().convert('RGB')
        open_cv_frame = np.array(open_cv_frame)

        h, w = np.shape(open_cv_frame)[:2]

        # Ground truth point is out of the frame
        index_to_exclude = []
        for p in range(len(lms_gt)):
            if lms_gt[p,0] >= h or lms_gt[p,1] >= w or lms_gt[p,1] <= 0 or lms_gt[p,0] <= 0:
                index_to_exclude.append(p)



        # Mempo works with (y,x) and not (x,y)!
        if pose == '0':
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

        elif pose == '30':
            eye_center = [(lms_gt[13,1] + lms_gt[10,1])/2, (lms_gt[12,0] + lms_gt[15,0])/2]
            #select the nose center
            nose_center = [(lms_gt[18,1] + lms_gt[16,1])/2, (lms_gt[20,0] + lms_gt[17,0])/2]
            #calculate the distance
            yard_stick = list(np.array(eye_center) - np.array(nose_center))
            norm = math.sqrt(yard_stick[0] **2 + yard_stick[1] ** 2)

        elif pose == '60':
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
        if save_images:
            gt_frame = open_cv_frame.copy()

            j = 0
            for (y, x) in lms_pred:
                if j in lms:
                    cv.circle(open_cv_frame, (int(x), int(y)), r, (255,0,0), thickness=-1, lineType=cv.LINE_AA)
                else:
                    cv.circle(open_cv_frame, (int(x), int(y)), r, (255,0,255), thickness=-1, lineType=cv.LINE_AA)
                j += 1

            file = files[i].split('/')[-1].split('.')[0]
            cv.imwrite(os.path.join(folder, '%s_%.3f.jpg' % (file,np.mean(plm_errors[-1]))), cv.cvtColor(open_cv_frame, cv.COLOR_BGR2RGB))
            #print(os.path.join(folder, '%s_%.3f_gt.jpg' % (file,np.mean(plm_errors[-1]))))
            for k in range(len(lms_pred)):
                cv.line(open_cv_frame, (int(lms_pred[k, 1]), int(lms_pred[k, 0])), (int(lms_gt[k, 1]), int(lms_gt[k, 0])), (255,255,255), thickness=2)

            cv.imwrite(os.path.join(folder, '%s_%.3f_lines.jpg' % (file,np.mean(plm_errors[-1]))), cv.cvtColor(open_cv_frame, cv.COLOR_BGR2RGB))


            j = 0
            for (y, x) in lms_gt:
                if j in lms:
                    cv.circle(gt_frame, (int(x), int(y)), r, (255,0,0), thickness=-1, lineType=cv.LINE_AA)
                else:
                    cv.circle(gt_frame, (int(x), int(y)), r, (255,0,255), thickness=-1, lineType=cv.LINE_AA)
                j += 1

            cv.imwrite(os.path.join(folder, '%s_gt.jpg' % file), cv.cvtColor(gt_frame, cv.COLOR_BGR2RGB))

                #print(os.path.join(folder, '%s_%.3f_gt.jpg' % (file,np.mean(plm_errors[-1]))))
            for k in range(len(lms_pred)):
                cv.line(open_cv_frame, (int(lms_pred[k, 1]), int(lms_pred[k, 0])), (int(lms_gt[k, 1]), int(lms_gt[k, 0])), (255,255,255), thickness=2)

            #cv.imwrite(os.path.join(folder, '%s_%.3f.jpg' % (file,np.mean(plm_errors[-1]))), cv.cvtColor(gt_frame, cv.COLOR_BGR2RGB))


    plm_errors = np.vstack(plm_errors)
    print(np.shape(plm_errors))
    SR = 0.06
    if pose == '0':


        ear_error = [e for e in np.hstack(plm_errors[:,:11]) if e != 0]
        print('Ears mean error: ', np.mean(ear_error))
        print('Ears success rate: ',len(np.where(np.hstack(ear_error) < SR)[0]) / len(np.hstack(ear_error)))

        nose_error =  [e for e in np.hstack(plm_errors[:,16:29]) if e != 0]
        print('Nose mean error: ', np.mean(nose_error))
        print('Nose success rate: ',len(np.where(np.hstack(nose_error) < SR)[0]) / len(np.hstack(nose_error)))

        eye_error = [e for e in np.hstack(plm_errors[:,10:17]) if e != 0]
        print('Eye mean error: ', np.mean(eye_error))
        print('Eye success rate: ',len(np.where(np.hstack(eye_error) < SR)[0]) / len(np.hstack(eye_error)))

        eye_error =  [e for e in np.hstack(plm_errors[:,28:35]) if e != 0]
        print('Second eye mean error: ', np.mean(eye_error))
        print('Second eye success rate: ',len(np.where(np.hstack(eye_error) < SR)[0]) / len(np.hstack(eye_error)))

    elif pose == '60':
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

    elif pose == '30':
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



    print('Mean error:         ', np.mean([e for e in np.hstack(plm_errors) if e != 0]))
    print('Less than 10% error:', len(np.where(np.hstack([e for e in np.hstack(plm_errors) if e != 0]) < SR) [0]) / len(np.hstack([e for e in np.hstack(plm_errors) if e != 0])))




    return plm_errors

def error_dist(errors, export=None):
    errors = sorted(errors)
    ys = []
    xs = np.arange(0, 1, 0.005)
    out_string = ''
    for x in xs:
        y = len(np.where(errors < x)[0]) / len(errors)
        ys.append(y)
        out_string += str(x) + ',' + str(y) + '\n'
    ys = np.array(ys)
    print('Mean error:         ', np.mean(errors))
    print('Less than 10% error:', len(np.where(np.array(errors) < 0.1)[0]) / len(errors))
    print('AUC:                ', np.mean(ys))
    if export is not None:
        with open(export + '.dat', 'w') as out_file:
            out_file.write(out_string)

    return (xs, ys)


def ERT(data, path_to_images, n_pert = 30, prefix = '', verbose = True):
    train = data[0][:]
    name_fitter = prefix + 'final.pkl'
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

    if MODE == 'cross_val':
        errors = []
        for k in range(len(data[0])):
            train = data[0][k][0]
            #train_lms= all_fold_data[1][k][0]
            val = data[0][k][1]
            #val_lms = all_fold_data[1][k][1]

            print('Starting fold', k + 1)
            name_fitter = prefix + 'fold_' + str(k + 1) + '_' + '.pkl'
            print('fitters/' + name_fitter)

            """
            Train or load the ERT model
            ===========================================
            """
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


            """
            Test the model
            ===========================================
            """
            if verbose:
                print('Validation fold', k + 1)
            mean_error, all_errors = test_eval(fitter, val)
            errors.extend(all_errors)

            print('Mean norm. error: '.ljust(25), mean_error)
            print('Success rate (error < 0.10: '.ljust(25), len(np.where(np.array(all_errors) < 0.1)[0]) / len(all_errors))
            print('Finished fold: ', k + 1)
            print()

        return errors

def new_SDM(data, path_to_images, n_pert = 30, prefix = '', verbose = True):
    if MODE == 'final_model':
        train = data[0][:]
        name_fitter = prefix + 'final.pkl'
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


    elif MODE == 'cross_val':
        errors = []
        for k in range(len(data[0])):
            train = data[0][k][0]
            #train_lms= _data[1][k][0]
            val = data[0][k][1]
            #val_lms = data[1][k][1]

            print('Starting fold', k + 1)
            name_fitter = prefix + 'fold_' + str(k + 1) + '_' + '.pkl'
            print('fitters/' + name_fitter)

            """
            Train or load the SDM model
            ===========================================
            """
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
            """
            Test the model
            ===========================================
            """
            if verbose:
                print('Validation fold', k + 1)
            mean_error, all_errors = test_eval(fitter, val)
            errors.extend(all_errors)

            print('Mean norm. error: '.ljust(25), mean_error)
            print('Success rate (error < 0.10: '.ljust(25), len(np.where(np.array(all_errors) < 0.1)[0]) / len(all_errors))
            print('Finished fold: ', k + 1)
            print()

        return errors

def fit_mean_shape(data, prefix, path_to_images, verbose = True):

    if MODE == 'cross_val':
        for k in range(len(data[0])):
            all_errors = []
            train = data[0][k][0]
            val = data[0][k][1]


            mean_shape = mean_pointcloud([image.landmarks['PTS']for image in train])
            if verbose:
                print('Validation fold', k + 1)

            for image in val:
                error = euclidean_bb_normalised_error(mean_shape, image.landmarks['PTS'])
                all_errors.append(error)
            mean_error = np.mean(all_errors)


            print('Mean norm. error: '.ljust(25), mean_error)
            print('Success rate (error < 0.10: '.ljust(25), len(np.where(np.array(all_errors) < 0.1)[0]) / len(all_errors))
            print('Finished fold: ', k + 1)
            print()
            return all_errors
    elif MODE == 'final_model':
        if os.path.exists(os.path.join(path_to_images, 'fitters')) is not True:
                os.mkdir(os.path.join(path_to_images, 'fitters'))
        train = data[0][:]
        name_fitter = prefix + 'final.pkl'
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


#%%============================================================================
#                                           RUN
#=============================================================================
def main():
    if MODE == 'cross_val':
        path_to_images = os.path.join(ABS_POSE,'0_30/')
        images_30, files_30 = sorted_image_import(path_to_images)
        landmarks_30 = images_30.map(lambda x: x.landmarks) #Extracts the landmarks (associated with each image)

        cv_images_30 = cross_validation_sets(images_30, files_30, verbose=True)
        cv_landmarks_30 = cross_validation_sets(landmarks_30, files_30)


        for n_pert in [70]:
            prefix = '0_30_pert_%d_' % n_pert
            ert_errors_30 = ERT((cv_images_30, cv_landmarks_30), path_to_images, n_pert= n_pert, prefix = ('ert_' + prefix), verbose = True)
            sdm_errors_30 = new_SDM((cv_images_30, cv_landmarks_30), path_to_images, n_pert=n_pert, prefix= ('sdm_' + prefix), verbose = True)

        #all_errors = fit_mean_shape((cv_images_30, cv_landmarks_30), verbose = True)

        path_to_images = os.path.join(ABS_POSE,'30_60/')
        images_60, files_60 = sorted_image_import(path_to_images)
        landmarks_60 = images_60.map(lambda x: x.landmarks) #Extracts the landmarks (associated with each image)

        cv_images_60 = cross_validation_sets(images_60, files_60, verbose=True)
        cv_landmarks_60 = cross_validation_sets(landmarks_60, files_60)

        for n_pert in [70]:
            prefix = '0_60_pert_%d_' % n_pert
            ert_errors_60 = ERT((cv_images_60, cv_landmarks_60), path_to_images, n_pert= n_pert, prefix = ('ert_' + prefix), verbose = True)
            sdm_errors_60 = new_SDM((cv_images_60, cv_landmarks_60), path_to_images, n_pert=n_pert, prefix= ('sdm_' + prefix), verbose = True)

        #all_errors = fit_mean_shape((cv_images_60, cv_landmarks_60), verbose = True)

        path_to_images = os.path.join(ABS_POSE,'60_90/')
        images_90, files_90 = sorted_image_import(path_to_images)
        landmarks_90 = images_90.map(lambda x: x.landmarks) #Extracts the landmarks (associated with each image)

        cv_images_90 = cross_validation_sets(images_90, files_90, verbose=True)
        cv_landmarks_90 = cross_validation_sets(landmarks_90, files_90)

        for n_pert in [70]:
            prefix = '0_90_pert_%d_' % n_pert
            ert_errors_90 = ERT((cv_images_90, cv_landmarks_90), path_to_images, n_pert= n_pert, prefix = ('ert_' + prefix), verbose = True)
            sdm_errors_90 = new_SDM((cv_images_90, cv_landmarks_90), path_to_images, n_pert=n_pert, prefix= ('sdm_' + prefix), verbose = True)

        #all_errors = fit_mean_shape((cv_images_30, cv_landmarks_30), verbose = True)

    elif MODE == 'final_model':

        n_pert = 30
        prefix = '0_30_pert_%d_' % n_pert
        path_to_images = os.path.join(ABS_POSE,'')
        images_30, files_30 = sorted_image_import(path_to_images)
        landmarks_30 = images_30.map(lambda x: x.landmarks) #Extracts the landmarks (associated with each image)

        #fit_mean_shape((images_30, landmarks_30), 'mean_' + prefix, path_to_images, verbose = True)

        #ERT((images_30, landmarks_30), path_to_images, n_pert= n_pert, prefix = ('ert_' + prefix), verbose = True)
        new_SDM((images_30, landmarks_30), path_to_images, n_pert=n_pert, prefix= ('sdm_' + prefix), verbose = True)


        n_pert = 10
        prefix = '30_60_pert_%d_' % n_pert
        path_to_images = os.path.join(ABS_POSE,'30_60/')
        images_60, files_60 = sorted_image_import(path_to_images)
        landmarks_60 = images_60.map(lambda x: x.landmarks) #Extracts the landmarks (associated with each image)

        #fit_mean_shape((images_60, landmarks_60), 'mean_' + prefix, path_to_images, verbose = True)

        #ERT((images_60, landmarks_60), path_to_images, n_pert= n_pert, prefix = ('ert_' + prefix), verbose = True)
        new_SDM((images_60, landmarks_60), path_to_images, n_pert=n_pert, prefix= ('sdm_' + prefix), verbose = True)

        n_pert = 30
        prefix = '60_90_pert_%d_' % n_pert
        path_to_images = os.path.join(ABS_POSE,'60_90/')
        images_90, files_90 = sorted_image_import(path_to_images)
        landmarks_90 = images_90.map(lambda x: x.landmarks) #Extracts the landmarks (associated with each image)

        #fit_mean_shape((images_90, landmarks_90), 'mean_' + prefix, path_to_images, verbose = True)

        #ERT((images_90, landmarks_90), path_to_images, n_pert= n_pert, prefix = ('ert_' + prefix), verbose = True)
        new_SDM((images_90, landmarks_90), path_to_images, n_pert=n_pert, prefix= ('sdm_' + prefix), verbose = True)


#main()