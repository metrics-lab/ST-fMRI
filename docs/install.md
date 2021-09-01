## Installation

We provide some advice for installation all dependencies with conda. 

### Prepare environment

1. create a conda environement

```
conda create -n ST-fMRI python=3.7
```

2. activate the environment

```
conda activate ST-fMRI
```

3. install pytorch dependencies

```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
```

Please check your CUDA version on </url>[PyTorch](https://pytorch.org/) to use appropriate command.

4. install requierements

```
conda install --file requirements.txt
```