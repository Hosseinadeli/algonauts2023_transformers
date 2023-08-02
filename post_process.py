# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from matplotlib import pyplot as plt

import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os

import numpy as np
import pandas as pd 
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model

from tqdm import tqdm

from utils import *
#from datasets.loaddata_g import *
from dataset_algonauts import fetch_dataloader

from scipy.stats import pearsonr as corr

import code


from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.linear_model import LinearRegression, Ridge

lh_corr = []
rh_corr = []

verbose = 1
plot_figs = 0

data_dir = '../../algonauts_2023_challenge_data/'

save_npy = 1
num_fmri_pcs = 70
num_model_pcs = 100

import sys
 
subject_ind = int(sys.argv[1])

for subj in [subject_ind]: # [1]: # 
    
    dis_file = f'/share/nklab/projects/natural_scenes_dataset/nsddata/ppdata/subj0{subj}/behav/responses.tsv'

    df = pd.read_csv(dis_file, sep='\t')
    
    df['73KID'] -= 1

    k_list = ['RUN', 'TRIAL', 'TIME', 'ISOLD',
           'ISCORRECT', 'RT', 'CHANGEMIND', 'MEMORYRECENT', 'MEMORYFIRST',
           'ISOLDCURRENT', 'ISCORRECTCURRENT', 'TOTAL1', 'TOTAL2', 'BUTTON',]

    
    class argObj:
        def __init__(self, data_dir, subj):

            self.subj = format(subj, '02')
            self.data_dir = os.path.join(data_dir, 'subj'+self.subj)

    args = argObj(data_dir, subj)
    
    train_img_dir  = os.path.join(args.data_dir, 'training_split', 'training_images')
    test_img_dir  = os.path.join(args.data_dir, 'test_split', 'test_images')

    # Create lists will all training and test image file names, sorted
    train_img_list = os.listdir(train_img_dir)
    train_img_list.sort()
    test_img_list = os.listdir(test_img_dir)
    test_img_list.sort()
#     print('Training images: ' + str(len(train_img_list)))
#     print('Test images: ' + str(len(test_img_list)))

    train_imgs_paths = sorted(list(Path(train_img_dir).iterdir()))
    test_imgs_paths = sorted(list(Path(test_img_dir).iterdir()))

    #idxs_train = np.arange(len(train_img_list))
    idxs_test = np.arange(len(test_img_list))
    
    
    ims_subj_train = []
    for i in range(len(train_imgs_paths)):

        ind = int(str(train_imgs_paths[i]).split('-')[-1].split('.')[0])
        ims_subj_train.append(ind)

    #fts_subj_train = np.hstack(fts_subj_train)

    ims_subj_test = []
    for i in range(len(test_imgs_paths)):

        ind = int(str(test_imgs_paths[i]).split('-')[-1].split('.')[0])
        ims_subj_test.append(ind)


    # Load the ROI classes mapping dictionaries
    roi_mapping_files = ['mapping_prf-visualrois.npy', 'mapping_floc-bodies.npy',
        'mapping_floc-faces.npy', 'mapping_floc-places.npy',
        'mapping_floc-words.npy', 'mapping_streams.npy']
    roi_name_maps = []
    for r in roi_mapping_files:
        roi_name_maps.append(np.load(os.path.join(args.data_dir, 'roi_masks', r),
            allow_pickle=True).item())

    # Load the ROI brain surface maps
    lh_challenge_roi_files = ['lh.prf-visualrois_challenge_space.npy',
        'lh.floc-bodies_challenge_space.npy', 'lh.floc-faces_challenge_space.npy',
        'lh.floc-places_challenge_space.npy', 'lh.floc-words_challenge_space.npy',
        'lh.streams_challenge_space.npy']
    rh_challenge_roi_files = ['rh.prf-visualrois_challenge_space.npy',
        'rh.floc-bodies_challenge_space.npy', 'rh.floc-faces_challenge_space.npy',
        'rh.floc-places_challenge_space.npy', 'rh.floc-words_challenge_space.npy',
        'rh.streams_challenge_space.npy']
    lh_challenge_rois = []
    rh_challenge_rois = []
    for r in range(len(lh_challenge_roi_files)):
        lh_challenge_rois.append(np.load(os.path.join(args.data_dir, 'roi_masks',
            lh_challenge_roi_files[r])))
        rh_challenge_rois.append(np.load(os.path.join(args.data_dir, 'roi_masks',
            rh_challenge_roi_files[r])))
                
    
    fmri_dir = os.path.join(args.data_dir, 'training_split', 'training_fmri')
    lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
    rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))
  
    
    
    for readout_res in ['streams_inc', 'visuals', 'bodies', 'places', 'faces', 'words']:
    
        if readout_res == 'streams_inc':
            selected_layers = [1,2,3,4,5,6]
        elif readout_res == 'visuals':
            selected_layers = [5,6,7,8,9]
        elif readout_res == 'bodies':
            selected_layers = [1,2,3,4]
        elif readout_res == 'places':
            selected_layers = [1,2,3,4]
        elif readout_res == 'faces':
            selected_layers = [4]
        elif readout_res == 'words':
            selected_layers = [3,4]    


        for lay in selected_layers: #selected_layers: # [1]: #[1,2,3,4,5,6]: # [5,6,7,8,9]: # range(5,13): # [6]: # range(5,11): detr_dino_10_bodies_16

            test_save_dir = '../results/detr_dino_'+ str(lay)+ '_' + readout_res + '_16/' + str(args.subj)   

            if not (os.path.exists(test_save_dir+ '/run10/lh_pred_test.npy')): continue 
            print(test_save_dir)

            lh_corr_runs = []
            rh_corr_runs = []
            
            lh_pred_corr = []
            rh_pred_corr = []

            lh_fmri_val_runs = []
            rh_fmri_val_runs = []

            lh_fmri_val_pred_runs = []
            rh_fmri_val_pred_runs = []

            idxs_val_runs = []

            for run in range(1,11):  #[1,2,3,4,5]: # range(1,7):

                #subj_res_dir = '../results/detr_dino_'+ str(lay)+ '_' + readout_res + '_16/'+ str(args.subj) + '/run' + str(run) + '/'

                subj_res_dir = test_save_dir + '/run' + str(run) + '/'

                print(subj_res_dir)
                if not (os.path.exists(subj_res_dir + '/idxs.npy')): continue 

                idxs = np.load(subj_res_dir + '/idxs.npy')  

                num_train = int(np.round(len(idxs) / 100 * 90))
                idxs_train, idxs_val = idxs[:num_train], idxs[num_train:]


                #     print('LH training fMRI data shape:')
                #     print(lh_fmri.shape)
                #     print('(Training stimulus images × LH vertices)')

                #     print('\nRH training fMRI data shape:')
                #     print(rh_fmri.shape)
                #     print('(Training stimulus images × RH vertices)')

                lh_fmri_train = lh_fmri[idxs_train]
                lh_fmri_val = lh_fmri[idxs_val]
                rh_fmri_train = rh_fmri[idxs_train]
                rh_fmri_val = rh_fmri[idxs_val]

                lh_fmri_val_pred = np.load(subj_res_dir + 'lh_fmri_val_pred.npy')
                rh_fmri_val_pred = np.load(subj_res_dir + 'rh_fmri_val_pred.npy')


                lh_fmri_val_runs.append(lh_fmri_val)
                rh_fmri_val_runs.append(rh_fmri_val)

                lh_fmri_val_pred_runs.append(lh_fmri_val_pred)
                rh_fmri_val_pred_runs.append(rh_fmri_val_pred)

                idxs_val_runs.append(idxs_val)



            
            lh_fmri_val_runs = np.concatenate(lh_fmri_val_runs, 0)
            rh_fmri_val_runs = np.concatenate(rh_fmri_val_runs, 0)

            lh_fmri_val_pred_runs = np.concatenate(lh_fmri_val_pred_runs, 0)
            rh_fmri_val_pred_runs = np.concatenate(rh_fmri_val_pred_runs, 0)

            idxs_val_runs = np.concatenate(idxs_val_runs, 0)

            beh_res = np.zeros((len(idxs_val_runs),len(k_list))).astype(float)

            for i in range(len(idxs_val_runs)):
                #print(np.array(ims_subj_train)[idxs_val_runs[i]])
                df_t = df[df['73KID'] == np.array(ims_subj_train)[idxs_val_runs[i]]]

                for k in range(len(k_list)):
                    beh_res[i,k] = np.nan_to_num(df_t[k_list[k]].to_numpy()).astype(float).mean()


            # fmri pca
            pca_fmri_lh = PCA(n_components=num_fmri_pcs)
            pca_fmri_lh.fit(lh_fmri_val_runs)

            lh_fmri_val_runs_pca = pca_fmri_lh.transform(lh_fmri_val_runs)

            pca_fmri_rh = PCA(n_components=num_fmri_pcs)
            pca_fmri_rh.fit(rh_fmri_val_runs)

            rh_fmri_val_runs_pca = pca_fmri_rh.transform(rh_fmri_val_runs)


            # model pca
            pca_lh = PCA(n_components=num_model_pcs)
            pca_lh.fit(lh_fmri_val_pred_runs)

            lh_fmri_val_pred_runs_pca = pca_lh.transform(lh_fmri_val_pred_runs)


            pca_rh = PCA(n_components=num_model_pcs)
            pca_rh.fit(rh_fmri_val_pred_runs)

            rh_fmri_val_pred_runs_pca = pca_rh.transform(rh_fmri_val_pred_runs)
     


            lh_pred_test = []
            rh_pred_test = []
                
            # repeat the process 30 times 
            for r in range(0,30):  #[1,2,3,4,5]: # range(1,7):
                # shuffle 

                idxs_runs = np.arange(len(idxs_val_runs))
                np.random.shuffle(idxs_runs)
                num_train = int(np.round(len(idxs_runs) / 100 * 90))
                idxs_runs_train, idxs_runs_val = idxs_runs[:num_train], idxs_runs[num_train:]

                beh_res_train = beh_res[idxs_runs_train]
                beh_res_val = beh_res[idxs_runs_val]

                lh_features_train = lh_fmri_val_pred_runs_pca[idxs_runs_train]
                lh_features_val = lh_fmri_val_pred_runs_pca[idxs_runs_val]

                rh_features_train = rh_fmri_val_pred_runs_pca[idxs_runs_train]
                rh_features_val = rh_fmri_val_pred_runs_pca[idxs_runs_val]
                

                # concatenate beh features
             
                rh_features_train = np.concatenate((rh_features_train,beh_res_train), 1)
                rh_features_val = np.concatenate((rh_features_val, beh_res_val), 1)
                
                lh_fmri_train = lh_fmri_val_runs_pca[idxs_runs_train]
                lh_fmri_val = lh_fmri_val_runs[idxs_runs_val]

                rh_fmri_train = rh_fmri_val_runs_pca[idxs_runs_train]
                rh_fmri_val = rh_fmri_val_runs[idxs_runs_val]


                # Fit linear regressions on the training data
                reg_lh = LinearRegression().fit(lh_features_train, lh_fmri_train)
                reg_rh = LinearRegression().fit(rh_features_train, rh_fmri_train)
                
#                 model_lh = Ridge(alpha=1)
#                 reg_lh = model_lh.fit(lh_features_train, lh_fmri_train)
#                 model_rh = Ridge(alpha=1)
#                 reg_rh = model_rh.fit(rh_features_train, rh_fmri_train)
                
                
                # Use fitted linear regressions to predict the validation and test fMRI data
                lh_fmri_val_pred = reg_lh.predict(lh_features_val)
                #     lh_fmri_test_pred = reg_lh.predict(features_test)
                rh_fmri_val_pred = reg_rh.predict(rh_features_val)
                #     rh_fmri_test_pred = reg_rh.predict(features_test)
                

                lh_fmri_val_pred = pca_fmri_lh.inverse_transform(lh_fmri_val_pred)
                lh_fmri_val_pred.shape

                rh_fmri_val_pred = pca_fmri_rh.inverse_transform(rh_fmri_val_pred)
                rh_fmri_val_pred.shape


                # Empty correlation array of shape: (LH vertices)
                lh_correlation = np.zeros(lh_fmri_val_pred.shape[1])
                # Correlate each predicted LH vertex with the corresponding ground truth vertex
                for v in tqdm(range(lh_fmri_val_pred.shape[1])):
                    lh_correlation[v] = corr(lh_fmri_val_pred[:,v], lh_fmri_val[:,v])[0]

                # Empty correlation array of shape: (RH vertices)
                rh_correlation = np.zeros(rh_fmri_val_pred.shape[1])
                # Correlate each predicted RH vertex with the corresponding ground truth vertex
                for v in tqdm(range(rh_fmri_val_pred.shape[1])):
                    rh_correlation[v] = corr(rh_fmri_val_pred[:,v], rh_fmri_val[:,v])[0]

                    
                print('rh_correlation.shape', rh_correlation.shape)

                # Select the correlation results vertices of each ROI
                roi_names = []
                lh_roi_correlation = []
                rh_roi_correlation = []
                for r1 in range(len(lh_challenge_rois)):
                    for r2 in roi_name_maps[r1].items():
                        if r2[0] != 0: # zeros indicate to vertices falling outside the ROI of interest
                            roi_names.append(r2[1])
                            lh_roi_idx = np.where(lh_challenge_rois[r1] == r2[0])[0]
                            rh_roi_idx = np.where(rh_challenge_rois[r1] == r2[0])[0]
                            lh_roi_correlation.append(lh_correlation[lh_roi_idx])
                            rh_roi_correlation.append(rh_correlation[rh_roi_idx])
                roi_names.append('All vertices')
                lh_roi_correlation.append(lh_correlation)
                rh_roi_correlation.append(rh_correlation)
                
                lh_pred_corr.append(lh_roi_correlation[-1])
                rh_pred_corr.append(rh_roi_correlation[-1])

                # Create the plot
                lh_mean_roi_correlation = [np.mean(np.nan_to_num(np.array(lh_roi_correlation[r]), copy=True, nan=0.0, posinf=None, neginf=None))
                    for r in range(len(lh_roi_correlation))]
                rh_mean_roi_correlation = [np.mean(np.nan_to_num(np.array(rh_roi_correlation[r]), copy=True, nan=0.0, posinf=None, neginf=None))
                    for r in range(len(rh_roi_correlation))]

#                 if plot_figs:
#                     plt.figure(figsize=(18,6))
#                     x = np.arange(len(roi_names))
#                     width = 0.30
#                     plt.bar(x - width/2, lh_mean_roi_correlation, width, label='Left Hemisphere')
#                     plt.bar(x + width/2, rh_mean_roi_correlation, width,
#                         label='Right Hemishpere')
#                     plt.xlim(left=min(x)-.5, right=max(x)+.5)
#                     plt.ylim(bottom=0, top=1)
#                     plt.xlabel('ROIs')
#                     plt.xticks(ticks=x, labels=roi_names, rotation=60)
#                     plt.ylabel('Mean Pearson\'s $r$')
#                     plt.legend(frameon=True, loc=1);
#                     plt.show()

                lh_corr_runs.append(lh_mean_roi_correlation[-1])
                rh_corr_runs.append(rh_mean_roi_correlation[-1])


                if verbose:
                    print(f'subj: {subj}   lh_corr: {lh_mean_roi_correlation[-1]}   rh_corr: {rh_mean_roi_correlation[-1]}')


                # apply to test 

                beh_test = np.zeros((len(idxs_test),len(k_list))).astype(float)

                for i in range(len(idxs_test)):
                    df_t = df[df['73KID'] == np.array(ims_subj_test)[idxs_test[i]]]

                    for k in range(len(k_list)):
                        beh_test[i,k] = np.nan_to_num(df_t[k_list[k]].to_numpy()).astype(float).mean()


                for run in range(1,11):

                    subj_res_dir = test_save_dir + '/run' + str(run) + '/'

                    lh_fmri_test_pred_m = np.load(subj_res_dir + 'lh_pred_test.npy')
                    rh_fmri_test_pred_m = np.load(subj_res_dir + 'rh_pred_test.npy')

                    lh_fmri_test_pred_m_pca = pca_lh.transform(lh_fmri_test_pred_m)
                    rh_fmri_test_pred_m_pca = pca_rh.transform(rh_fmri_test_pred_m)

                    lh_features_test = lh_fmri_test_pred_m_pca # np.concatenate((lh_fmri_test_pred_m_pca, beh_test), 1)
                    rh_features_test = rh_fmri_test_pred_m_pca # np.concatenate((rh_fmri_test_pred_m_pca, beh_test), 1)
                  
                    lh_fmri_test_pred_pca = reg_lh.predict(lh_features_test)
                    rh_fmri_test_pred_pca = reg_rh.predict(rh_features_test)

                    lh_fmri_test_pred = pca_fmri_lh.inverse_transform(lh_fmri_test_pred_pca)
                    rh_fmri_test_pred = pca_fmri_rh.inverse_transform(rh_fmri_test_pred_pca)

                    lh_pred_test.append(lh_fmri_test_pred.astype(np.float32))
                    rh_pred_test.append(rh_fmri_test_pred.astype(np.float32))


            #print(np.array(lh_pred_test).shape)
            lh_pred_test = np.array(lh_pred_test).mean(0)
            rh_pred_test = np.array(rh_pred_test).mean(0)
            
            #print(np.array(lh_pred_corr).shape)
            lh_pred_corr = np.array(lh_pred_corr).mean(0)
            rh_pred_corr = np.array(rh_pred_corr).mean(0)
            

            # Select the correlation results vertices of each ROI
            roi_names = []
            lh_roi_correlation = []
            rh_roi_correlation = []
            for r1 in range(len(lh_challenge_rois)):
                for r2 in roi_name_maps[r1].items():
                    if r2[0] != 0: # zeros indicate to vertices falling outside the ROI of interest
                        roi_names.append(r2[1])
                        lh_roi_idx = np.where(lh_challenge_rois[r1] == r2[0])[0]
                        rh_roi_idx = np.where(rh_challenge_rois[r1] == r2[0])[0]
                        lh_roi_correlation.append(lh_pred_corr[lh_roi_idx])
                        rh_roi_correlation.append(rh_pred_corr[rh_roi_idx])
            roi_names.append('All vertices')
            lh_roi_correlation.append(lh_correlation)
            rh_roi_correlation.append(rh_correlation)

            # Create the plot
            lh_mean_roi_correlation = [np.mean(np.nan_to_num(np.array(lh_roi_correlation[r]), copy=True, nan=0.0, posinf=None, neginf=None))
                for r in range(len(lh_roi_correlation))]
            rh_mean_roi_correlation = [np.mean(np.nan_to_num(np.array(rh_roi_correlation[r]), copy=True, nan=0.0, posinf=None, neginf=None))
                for r in range(len(rh_roi_correlation))]

            if plot_figs:
                plt.figure(figsize=(18,6))
                x = np.arange(len(roi_names))
                width = 0.30
                plt.bar(x - width/2, lh_mean_roi_correlation, width, label='Left Hemisphere')
                plt.bar(x + width/2, rh_mean_roi_correlation, width,
                    label='Right Hemishpere')
                plt.xlim(left=min(x)-.5, right=max(x)+.5)
                plt.ylim(bottom=0, top=1)
                plt.xlabel('ROIs')
                plt.xticks(ticks=x, labels=roi_names, rotation=60)
                plt.ylabel('Mean Pearson\'s $r$')
                plt.legend(frameon=True, loc=1);
                plt.show()
                    
                    
            if verbose:
                print(f'subj: {subj}   lh_corr: {lh_mean_roi_correlation[-1]}   rh_corr: {rh_mean_roi_correlation[-1]}')        
            

            if save_npy:
                np.save(test_save_dir+'/lh_pred_test_f70_m30.npy', lh_pred_test)
                np.save(test_save_dir+'/rh_pred_test_f70_m30.npy', rh_pred_test)
                
                np.save( test_save_dir + '/lh_roi_correlation_f70_m30.npy', lh_pred_corr)
                np.save( test_save_dir + '/rh_roi_correlation_f70_m30.npy', rh_pred_corr)


                #     test_save_dir = '../results/challenge_submission/tm_sm10_100_13/subj' + args.subj 
                #     if not os.path.exists(test_save_dir):
                #         os.makedirs(test_save_dir)

                #     np.save(test_save_dir+'/lh_pred_test.npy', lh_pred_test)
                #     np.save(test_save_dir+'/rh_pred_test.npy', rh_pred_test)


            lh_corr.append(lh_corr_runs)
            rh_corr.append(rh_corr_runs)

    # lh_corr = np.array(lh_corr)
    # rh_corr = np.array(rh_corr)

    # if verbose:
    #     print(f'lh_corr_mean: {lh_corr.mean()}   rh_corr_mean: {rh_corr.mean()}')
