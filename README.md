# InterLabelEC

InterLabelEC predict a query protein's catalytic activity, in the form of Enzyme Commision (EC) numbers. InterLabelEC uses the last three layers the [ESM2](https://github.com/facebookresearch/esm) large language model to extract sequence features, which are then learned by a series of neural networks to predict EC numbers under a new loss function that incorporates label imbalances and inter-label dependencies. 

## System Requirements

InterLabelEC is developed under a Linux environment with the following software:

- Python 3.11.5
- CUDA 12.1
- cuDNN 8.9.6
- DIAMOND v2.1.8
- NVIDIA drivers v.535.129.03

Python packages are detailed separately in `requirements.txt`.

## Set up InterLabelEC

3. Run `setup_env.sh` to create an environment and download ESM2 models

Alternatively, you can create the environment manually:

```bash
pip install -r requirements.txt
```

Then download ESM2 models:

```bash
this_file_path=$(dirname $(readlink -f $0))
mode_pt_url="https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt"
regression_url="https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t36_3B_UR50D-contact-regression.pt"
save_path=$this_file_path/Data/esm_models
mkdir -p $save_path
wget -P $save_path $mode_pt_url
wget -P $save_path $regression_url
```

This will download the ESM2 models into `Data/esm_models`.

To convert the format of the current EC database, run:

```bash
./conda/bin/python update_data.py
```

This will convert the EC database into the required format.

## Data Processing

Run the following command:

```bash
./conda/bin/python prepare_data.py --ia
```

This will:
- Convert raw data (`train_terms.tsv` and `train_seq.fasta`) into required training data
- Create an Information Content file for the training data (`--ia`)
- Extract the ESM embeddings for the training data

All paths are specified in `settings.py`.

## Model Usage

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

Quancheng Liu, Chengxin Zhang, Lydia Freddolino (2024)
[InterLabelGO+: unraveling label correlations in protein function prediction](https://doi.org/10.1093/bioinformatics/btae655)
Bioinformatics, 40(11): btae655.

