# -*- coding: utf-8 -*-
# @Author: Simon Dahan @SD3004
# @Date:   2021-08-26 15:02:02
# @Last Modified by:   Simon Dahan @SD3004
# @Last Modified time: 2021-08-27 15:13:00

import os
import argparse

import numpy as np
import pandas as pd

from scipy import stats
from sklearn.model_selection import StratifiedKFold


def main(args):

    ### loading subject list

    print('')
    print('#'*30)
    print('Starting: preprocessing script')
    print('#'*30)
    print('')

    if args.filename.find('txt') != -1:
        ids = np.loadtxt(args.filename)
        nb_subject = ids.shape[0]
    elif args.filename.find('csv') != -1: 
        ids = pd.read_csv(args.filename)
        nb_subject = ids.shape[0]
    else:
        raise TypeError('filetype not implemented')


    ICA_nodes = int(args.ICA)
    data_path = str(args.data_path)
    
    T = 1200
    data = np.zeros((nb_subject,1,T,ICA_nodes,1))
    label = np.zeros((nb_subject,))

    print('')
    print('Number of subjects: {}'.format(nb_subject))
    print('ICA nodes: {}'.format(ICA_nodes))
    print('Temporal length sequence: {}'.format(T))
    print('')

    ### loading node timeseries

    print('')
    print('#'*30)
    print('Loading: node timeseries')
    print('#'*30)
    print('')

    idx = 0
    non_used = []
    for i in range(nb_subject):

        subject_string = format(int(ids[i,0]),'06d')
        filename = 'HCP_PTN1200/node_timeseries/node_timeseries_{}/3T_HCP1200_MSMAll_d{}_ts2/'.format(ICA_nodes, ICA_nodes)+subject_string+'.txt'
        filepath = os.path.join(str(data_path),filename)

        if os.path.exists(filepath):
            full_sequence = pd.read_csv(filepath, delimiter = " ").to_numpy()[:1200,:]  
            if full_sequence.shape[0] < T:
                print('sequence too short :{}'.format(filepath))
                continue

            full_sequence = np.transpose(full_sequence[:T,:])

            z_sequence = stats.zscore(full_sequence, axis=1)
            
            if idx ==0:
                data_all = z_sequence
            else:
                data_all = np.concatenate((data_all, z_sequence), axis=1)
            
            data[idx,0,:,:,0] = np.transpose(z_sequence)
            label[idx] = ids[i,1]
            idx = idx + 1
            if idx%100==0:
                print('subject preprocessed: {}'.format(idx))
            
        else:
            non_used.append(subject_string)
    
    ### keep only subjects
    data = data[:idx,0,:,:,0]
    label = label[:idx]

    print('')
    print('Number of subjects loaded: {}'.format(idx))
    print('Data shape: {}'.format(data.shape))
    print('')

    df = pd.DataFrame(data=np.array(non_used))

    try:
        os.mkdir(args.output_folder)
        print('Creating folder: {}'.format(args.output_folder))
    except OSError:
        print('folder already exist: {}'.format(args.output_folder))

    df.to_csv(os.path.join(args.output_folder,'ids_not_used.csv'))

    print('')
    print('#'*30)
    print('Computing: Adjacency Matrix')
    print('#'*30)
    print('')
           
    
    # compute adj matrix
    n_regions = ICA_nodes
    A = np.zeros((n_regions, n_regions))
    for i in range(n_regions):
        for j in range(i, n_regions):
            if i==j:
                A[i][j] = 1
            else:
                A[i][j] = abs(np.corrcoef(data_all[i,:], data_all[j,:])[0][1]) # get value from corrcoef matrix
                A[j][i] = A[i][j]
    

    print('')
    print('saving adjacency matrix')
    print('')

    try:        
        os.mkdir(os.path.join(args.output_folder,'node_timeseries'))  
        print('Creating folder: {}'.format(os.path.join(args.output_folder,'node_timeseries')))
    except OSError:
        print('folder already exist: {}'.format(os.path.join(args.output_folder,'node_timeseries')))


    try:        
        os.mkdir(os.path.join(args.output_folder,'node_timeseries/node_timeseries_{}'.format(ICA_nodes)))  
        print('Creating folder: {}'.format(os.path.join(args.output_folder,'node_timeseries/node_timeseries_{}/'.format(ICA_nodes))))
    except OSError:
        print('folder already exist: {}'.format(os.path.join(args.output_folder,'node_timeseries/node_timeseries_{}/'.format(ICA_nodes))))

    np.save(os.path.join(args.output_folder,'node_timeseries/node_timeseries_{}/adj_matrix.npy'.format(ICA_nodes)), A)

    # split train/test and save data
    
    skf = StratifiedKFold(n_splits=5,shuffle=True)
    fold = 1
    for train_idx, test_idx in skf.split(data, label):
        train_data = data[train_idx]
        train_label = label[train_idx]
        test_data = data[test_idx]
        test_label = label[test_idx] 

        filename = os.path.join(args.output_folder,'node_timeseries/node_timeseries_{}'.format(ICA_nodes), 'train_data_1200_'+str(fold)+'.npy')
        np.save(filename,train_data)
        filename = os.path.join(args.output_folder,'node_timeseries/node_timeseries_{}'.format(ICA_nodes),'train_label_1200_'+str(fold)+'.npy')
        np.save(filename,train_label)
        filename = os.path.join(args.output_folder,'node_timeseries/node_timeseries_{}'.format(ICA_nodes),'test_data_1200_'+str(fold)+'.npy')
        np.save(filename,test_data)
        filename = os.path.join(args.output_folder,'node_timeseries/node_timeseries_{}'.format(ICA_nodes),'test_label_1200_'+str(fold)+'.npy')
        np.save(filename,test_label)

        fold = fold + 1
  


if __name__ == '__main__':

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Preprocessing script for ICA nodes timeseries')
    parser.add_argument('filename',help='list of HCP ids')
    parser.add_argument('ICA', help='number of ICA nodes')
    parser.add_argument('data_path', help='path to data')
    parser.add_argument('output_folder', help='path to save data')
     
    args = parser.parse_args()
    # Call training
    main(args)