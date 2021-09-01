#
# Created on Wed Sep 01 2021
#
# by Simon Dahan @SD3004
#
# Copyright (c) 2021 MeTrICS Lab
#

import os
import sys
sys.path.append('./')
sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.MSG3D.msgcn import MultiScale_GraphConv as MS_GCN
from models.MSG3D.mstcn import MultiScale_TemporalConv as MS_TCN
from models.MSG3D.msg3d import MultiWindow_MS_G3D



class Model(nn.Module):
    def __init__(self,
                 num_class,
                 num_nodes,
                 num_person,
                 num_gcn_scales,
                 num_g3d_scales,
                 path_to_data,
                 dropout=0,
                 in_channels=1):

        super(Model, self).__init__()

        A = np.load(os.path.join(path_to_data,'adj_matrix.npy'))

        A = A - np.eye(len(A), dtype=A.dtype)

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_nodes)

        print('Using dropout: {}'.format(dropout))

        # channels
        c1 = 96
        c2 = c1 * 2     # 192
        c3 = c2 * 2     # 384

        # r=3 STGC blocks
        self.gcn3d1 = MultiWindow_MS_G3D(in_channels, c1, A, num_g3d_scales,dropout, window_stride=1)
        self.sgcn1 = nn.Sequential(
            MS_GCN(num_gcn_scales, in_channels, c1, A, disentangled_agg=True),
            MS_TCN(c1, c1),
            MS_TCN(c1, c1))
        self.sgcn1[-1].act = nn.Identity()
        self.tcn1 = MS_TCN(c1, c1)

        self.gcn3d2 = MultiWindow_MS_G3D(c1, c2, A, num_g3d_scales,dropout,  window_stride=2)
        self.sgcn2 = nn.Sequential(
            MS_GCN(num_gcn_scales, c1, c1, A, disentangled_agg=True),
            MS_TCN(c1, c2, stride=2),
            MS_TCN(c2, c2))
        self.sgcn2[-1].act = nn.Identity()
        self.tcn2 = MS_TCN(c2, c2)

        self.gcn3d3 = MultiWindow_MS_G3D(c2, c3, A, num_g3d_scales, dropout, window_stride=2)
        self.sgcn3 = nn.Sequential(
            MS_GCN(num_gcn_scales, c2, c2, A, disentangled_agg=True),
            MS_TCN(c2, c3, stride=2),
            MS_TCN(c3, c3))
        self.sgcn3[-1].act = nn.Identity()
        self.tcn3 = MS_TCN(c3, c3)

        self.fc = nn.Linear(c3, num_class)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N * M, V, C, T).permute(0,2,3,1).contiguous()

        # Apply activation to the sum of the pathways
        x = F.relu(self.sgcn1(x) + self.gcn3d1(x), inplace=True)
        x = self.tcn1(x)

        x = F.relu(self.sgcn2(x) + self.gcn3d2(x), inplace=True)
        x = self.tcn2(x)
 
        x = F.relu(self.sgcn3(x) + self.gcn3d3(x), inplace=True)
        x = self.tcn3(x)

        out = x
        out_channels = out.size(1)
        out = out.view(N, M, out_channels, -1)
        out = out.mean(3)   # Global Average Pooling (Spatial+Temporal)
        out = out.mean(1)   # Average pool number of bodies in the sequence

        out = self.fc(out)
        return out

