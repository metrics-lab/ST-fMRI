# ST-fMRI
This repository contains PyTorch code for spatio-temporal deep learning on functional MRI data for phenotyping prediction. The original work was published at </url>[MLCN 2021](https://mlcnws.com/):


</url>[Improving Phenotype Prediction using Long-Range Spatio-Temporal Dynamics of Functional Connectivity](https://arxiv.org/abs/2109.03115)


## Downloading HCP Dataset

HCP data can be directly downloaded from </url>[Human Connectome Project](https://db.humanconnectome.org/)

## Installation

For PyTorch and dependencies installation, please follow instructions in [install.md](docs/install.md)

## Preprocessing 

In the folder /data/: 

```
python preprocessing_nodetimeseries.py  subjects.txt 25 /data/HCP/rfMRI ../outputs/
```


## Training Brain-MS-G3D 

For sex classification

```
python ./tools/train_node_timeseries.py --nodes 25 --bs 64 --epochs 100 --gpu 0 --windows 100 --data_path path_to_data
```

For fluid intelligence regression


```
python ./tools/train_node_timeseries.py --nodes 25 --bs 64 --epochs 100 --gpu 0 --windows 100 --fluid --data_path path_to_data
```

## Tensorboard

Starting tensorboard visualisation

```
tensorboard --logdir ./logs/MS-G3D/
```


## Docker support 

**Coming soon**

## References 

This repository is based on the following repositories:


- repository: </url>[MS-G3D](https://github.com/kenziyuliu/MS-G3D) - paper: </url>[Z.Liu et al 2020](https://arxiv.org/abs/2003.14111)


- repository: </url>[ST-GCN-HCP](https://github.com/sgadgil6/cnslab_fmri) - paper: </url>[S.Gadgil et al 2020](https://arxiv.org/abs/2003.10613)

and 

- repository: </url>[ST-GCN](https://github.com/yysijie/st-gcn) - paper: </url>[S.Yan et al 2018](https://arxiv.org/abs/1801.07455)


## Citation

Please cite this work if you found it useful:

```
@misc{dahan2021improving,
      title={Improving Phenotype Prediction using Long-Range Spatio-Temporal Dynamics of Functional Connectivity}, 
      author={Simon Dahan and Logan Z. J. Williams and Daniel Rueckert and Emma C. Robinson},
      year={2021},
      eprint={2109.03115},
      archivePrefix={arXiv},
      primaryClass={q-bio.NC}
      }
```

