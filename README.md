# MFE-DDI: A multi-view feature encoding framework for drug-drug interaction prediction


## Setup

A conda environment can be created with

`conda create --name MFE-DDI python=3.8`

`conda activate MFE-DDI`

`conda update -n base conda`

`conda env create -f environment.yaml`

## dataset

The dataset contains .zip file, please unzip it before training and predicting:

`unzip dataset/sdf.zip`

## Training

To train of the models run:

`python train.py`

## Predicting

To prediction of the models run:

`python predict.py`



