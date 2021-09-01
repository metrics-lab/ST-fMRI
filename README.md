# ST-fMRI
This repository contains code for spatio-temporal deep learning on functional MRI data

**COMING SOON...**


## Downloading HCP Dataset

HCP data can be directly downloaded from </url>https://db.humanconnectome.org/app/template/Login.vm


## Training Brain-MS-G3D 

For sex classification

`python ./tools/train_node_timeseries.py --nodes 25 --bs 64 --epochs 100 --gpu 0 --windows 100 --data_path path_to_data`

For fluid intelligence regression


`python train_node_timeseries.py --nodes 25 --bs 64 --epochs 100 --gpu 0 --windows 100 --fluid --data_path path_to_data`

## Tensorboard support

Starting tensorboard

`tensorboard --logdir ./logs/MS-G3D/`


## Docker support 

**Coming soon**

## References 

This repository is mostly based on:

-  </url>https://github.com/kenziyuliu/MS-G3D

and 

- </url>https://github.com/sgadgil6/cnslab_fmri


## Paper


## References