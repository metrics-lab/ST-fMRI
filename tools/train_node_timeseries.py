# -*- coding: utf-8 -*-
# @Author: Simon Dahan @SD3004
# @Date:   2021-08-27 15:21:46
# @Last Modified by:   Simon Dahan @SD3004
# @Last Modified time: 2021-08-31 16:57:25

import os
import argparse
import yaml
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from scipy.stats import pearsonr


from torch.utils.tensorboard import SummaryWriter



def train(args):

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print(device)   

    ICA_nodes = args.nodes
    num_epochs = args.epochs + 1
    TS = args.TS

    batch_size_training = args.bs 
    batch_size_testing = 64
    dropout = args.dropout

    num_scales_gcn = args.gcn_scales
    num_scales_g3d = args.g3d_scales

    LR = args.LR

    W_list = args.windows

    data_path = os.path.join(args.data_path,'node_timeseries/node_timeseries_{}'.format(ICA_nodes))


    hparams = {}
    hparams['bs'] = args.bs
    hparams['nodes'] = args.nodes
    hparams['epochs'] = args.epochs + 1
    hparams['windows'] = args.windows
    hparams['windows'] = args.TS
    hparams['g3d_scales'] = args.gcn_scales
    hparams['gcn_scales'] = args.g3d_scales
    hparams['dropout'] = args.dropout
    hparams['LR'] = args.LR
    hparams['optim'] = args.optim

    
    if args.fluid:
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCELoss()

    print('')
    print('#'*30)
    print('Logging training information')
    print('#'*30)
    print('')




    for ws in W_list:
        
        final_testing_accuracy = 0
        testing_acc_curr_fold = []
        
        ##############################
        ######      LOGGING     ######
        ##############################

        # creating folders for logging. 

        folder_to_save_model = '../logs/MS-G3D/'
        try:
            os.mkdir(folder_to_save_model)
            print('Creating folder: {}'.format(folder_to_save_model))
        except OSError:
            print('folder already exist: {}'.format(folder_to_save_model))

        # folder ICAs
        folder_to_save_model = os.path.join(folder_to_save_model,'ICA_{}'.format(ICA_nodes))
        try:
            os.mkdir(folder_to_save_model)
            print('Creating folder: {}'.format(folder_to_save_model))
        except OSError:
            print('folder already exist: {}'.format(folder_to_save_model))
        
        # folder ws
        folder_to_save_model = os.path.join(folder_to_save_model,'ws_{}'.format(ws))
        try:
            os.mkdir(folder_to_save_model)
            print('Creating folder: {}'.format(folder_to_save_model))
        except OSError:
            print('folder already exist: {}'.format(folder_to_save_model))


        date = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

        # folder time
        folder_to_save_model = os.path.join(folder_to_save_model,date)
        try:
            os.mkdir(folder_to_save_model)
            print('Creating folder: {}'.format(folder_to_save_model))
        except OSError:
            print('folder already exist: {}'.format(folder_to_save_model))

        # saving hparams
        
        with open(os.path.join(folder_to_save_model,'hparams.yaml'), 'w') as file:
            documents = yaml.dump(hparams, file)

        # starting tensorboard logging

        writer = SummaryWriter(log_dir=folder_to_save_model)

        ##############################
        ######     TRAINING     ######
        ##############################

        print('')
        print('#'*30)
        print('Starting training')
        print('#'*30)
        print('')

        print('-'*80)
        print("Window Size {}".format(ws))
        print('-'*80)
        for fold in range(1, 6):
            print('-'*80)
            print("Window Size {}, Fold {}".format(ws, fold))
            print('-'*80)
            
            best_test_acc_curr_fold = 0
            best_test_epoch_curr_fold = 0
            
            train_data = np.load(os.path.join(data_path,'train_data_1200_'+str(fold)+'.npy'))
            train_label = np.load(os.path.join(data_path,'train_label_1200_'+str(fold)+'.npy'))
            test_data = np.load(os.path.join(data_path,'test_data_1200_'+str(fold)+'.npy'))
            test_label = np.load(os.path.join(data_path,'test_label_1200_'+str(fold)+'.npy'))

            print(train_data.shape)
            
            net = msg3d.Model(num_class = 1,
                    num_point = ICA_nodes,
                    num_person = 1,
                    num_gcn_scales = num_scales_gcn,
                    num_g3d_scales = num_scales_g3d,
                    graph ='',
                    dropout= dropout)
        
            net.to(device)

            if args.optim == 'SGD':
                optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
            else:
                optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=0.001)

            

            for epoch in range(num_epochs):  # number of mini-batches
                # select a random sub-set of subjects
                idx_batch = np.random.permutation(int(train_data.shape[0]))
                idx_batch = idx_batch[:int(batch_size_training)]
        
                # construct a mini-batch by sampling a window W for each subject
                train_data_batch = np.zeros((batch_size_training, 1, ws, ICA_nodes, 1))
                train_label_batch = train_label[idx_batch]

                for i in range(batch_size_training):
                    r1 = random.randint(0, train_data.shape[2] - ws)
                    train_data_batch[i] = train_data[idx_batch[i], :, r1:r1 + ws, :, :]

                train_data_batch_dev = torch.from_numpy(train_data_batch).float().to(device)
                train_label_batch_dev = torch.from_numpy(train_label_batch).float().to(device)

                # forward + backward + optimize
                optimizer.zero_grad()

                outputs = net(train_data_batch_dev)
                if args.fluid:

                    loss = criterion(outputs.squeeze(), train_label_batch_dev)
                    writer.add_scalar('loss/train_{}'.format(fold), loss.item(), epoch+1)
                    correlation = pearsonr(outputs.data.cpu().numpy().reshape(-1),train_label_batch_dev.cpu())[0]
                    writer.add_scalar('correlation/train_{}'.format(fold), correlation, epoch + 1)


                else:

                    outputs = torch.nn.functional.sigmoid(outputs)
                    loss = criterion(outputs.squeeze(), train_label_batch_dev)
                    writer.add_scalar('loss/train_{}'.format(fold), loss.item(), epoch+1)
                    outputs = outputs.data.cpu().numpy() > 0.5
                    train_acc = sum(outputs[:, 0] == train_label_batch) / train_label_batch.shape[0]
                    writer.add_scalar('accuracy/train_{}'.format(fold), train_acc, epoch+1)

                loss.backward()
                optimizer.step()

                ##############################
                ######    VALIDATION    ######
                ##############################

                import pdb;pdb.set_trace()
            


        return 0 




if __name__ == '__main__':

    # Set up argument parser
        
    parser = argparse.ArgumentParser(description='MS-G3D')

    parser.add_argument(
                        '--nodes',
                        type=int,
                        required=True,
                        help='the number of ICA nodes')

    parser.add_argument(
                        '--bs',
                        type=int,
                        required=True,
                        help='batch size training')

    parser.add_argument(
                        '--epochs',
                        type=int,
                        required=True,
                        help='number of iterations')

    parser.add_argument(
                        '--gpu',
                        type=int,
                        required=True,
                        help='gpu to use')

    parser.add_argument(
                        '--data_path',
                        type=str,
                        required=True,
                        default='',
                        help='path where the data is stored')
                        
    parser.add_argument(
                        '--windows',
                        type=int,
                        required=True,
                        nargs='+',
                        help='windows')

    parser.add_argument(
                        '--dropout',
                        type=float,
                        required=False,
                        default=0,
                        help='windows')

                            
    parser.add_argument(
                        '--gcn_scales',
                        type=int,
                        required=False,
                        default=8,
                        help='windows')

                            
    parser.add_argument(
                        '--g3d_scales',
                        type=int,
                        required=False,
                        default=8,
                        help='windows')
        
    parser.add_argument(
                        '--LR',
                        type=float,
                        required=False,
                        default=0.001,
                        help='windows')

    parser.add_argument(
                        '--optim',
                        type=str,
                        required=False,
                        default='Adam',
                        help='windows')
    
    parser.add_argument(
                        '--TS',
                        type=int,
                        required=False,
                        default=64,
                        help='number of voters per test subject')

    parser.add_argument('--fluid',
                          action='store_true')

     
    args = parser.parse_args()
    # Call training
    train(args)