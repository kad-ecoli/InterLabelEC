# EZpred

EZpred predicts a query protein's catalytic activity, in the form of Enzyme Commision (EC) numbers. EZpred uses the last three layers the [ESM C](https://github.com/evolutionaryscale/esm?tab=readme-ov-file#esm-c-) large language model to extract sequence features, which are then learned by a series of neural networks to predict EC numbers under a new loss function that incorporates label imbalances and inter-label dependencies. 

## System Requirements

InterLabelEC is developed under a Linux environment with the following software:

- Python 3.11.5
- CUDA 12.1
- cuDNN 8.9.6
- DIAMOND v2.1.8
- NVIDIA drivers v.535.129.03

Python packages are detailed separately in `requirements.txt`.

## Set up InterLabelEC

Run `setup_env.sh` to create an environment and download ESM2 models

Alternatively, you can create the environment manually:

```bash
pip install -r requirements.txt
```

To convert the format of the current EC database, run:

```bash
./conda/bin/python update_data.py
```

This will convert the EC database into the required format.

## Data Processing

Download ESM C model:

```bash
wget "https://zenodo.org/records/15792215/files/Data.zip?download=1" -O Data.zip
unzip Data.zip
rm Data.zip
```

Run the following command:

```bash
./conda/bin/python prepare_data.py
```

Altenatively, you may use pre-curated data:

```bash
wget "https://zenodo.org/records/15812849/files/Data2.zip?download=1" -O Data2.zip
unzip Data2.zip
rm Data2.zip
```

This will:
- Convert raw data (`train_terms.tsv` and `train_seq.fasta`) into required training data
- Create an Information Content file for the training data (`--ia`)
- Extract the ESM embeddings for the training data

All paths are specified in `settings.py`.

## Model Usage

Download pre-trained models from [Zenodo](https://zenodo.org/records/15792215/files/models.zip?download=1)
```bash
wget "https://zenodo.org/records/15792215/files/models.zip?download=1" -O models.zip
unzip models.zip
rm models.zip
```

### 1. Prediction

```bash
./conda/bin/python predict.py -w example -f example/seq.fasta --use_gpu
```

This will predict EC numbers for the example sequence.

### 2. Retrain Models

```bash
./conda/bin/python train.py
```

Training configuration is specified in `settings.py`.

## Notes

1. The data processing code will overwrite the original data, and the training code will overwrite the original model.

## Citation

Chengxin Zhang, Quancheng Liu, Lydia Freddolino (2025)
[EZpred: improving deep learning-based enzyme function prediction using unlabeled sequence homologs](https://doi.org/10.1101/2025.07.09.663945)
bioRxiv

