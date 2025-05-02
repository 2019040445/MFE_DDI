# MFE-DDI: A multi-view feature encoding framework for drug-drug interaction prediction


## Setup

A conda environment can be created with

`conda create --name MFE-DDI python=3.8`

`conda activate MFE-DDI`

`conda update -n base conda`

`conda env create -f environment.yaml`

## dataset

### Pang et al.

- dataset/train.csv
- dataset/val.csv
- dataset/test.csv

### BioSNAP

- dataset/train_biosnap_smiles_new.csv
- dataset/test_biosnap_smiles_new.csv

### AdverseDDI

- dataset/AdverseDDI_train.csv
- dataset/AdverseDDI_valid.csv
- dataset/AdverseDDI_test.csv

## Training

To train of the models run:

`python train.py`

## Predicting

We update the well-trained weights on BioSNAP dataset. To prediction of the models run:

`python predict.py`
