#!/bin/bash

# Define the base directory for Conda installation
basedir=$(dirname $(readlink -f $0))
conda_dir="$basedir/conda"

# Check if Conda is installed; if not, download and install it
if [ ! -d "$conda_dir" ]; then
  echo "Conda not found. Installing Miniconda..."
  if [ ! -s "$basedir/miniconda.sh" ];then
      wget https://repo.anaconda.com/miniconda/Miniconda3-py311_24.11.1-0-Linux-x86_64.sh -O $basedir/miniconda.sh
  fi
  bash $basedir/miniconda.sh -b -p $conda_dir
  rm $basedir/miniconda.sh
fi

# Initialize Conda
#conda_path="$conda_dir"
#source "$conda_path/etc/profile.d/conda.sh"

this_file_path=$(dirname $(readlink -f $0))

# Create Conda environment
#conda create -n InterLabelGO python=3.11.5 -y
#conda activate InterLabelGO
#pip install -r $this_file_path/requirements.txt

$conda_dir/bin/pip install -r $this_file_path/requirements.txt
#$conda_dir/bin/pip install flash-attn --no-build-isolation


exit
# Download esmc_600m model if not already downloaded
mode_pt_url="https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12/resolve/main/data/weights/esmc_600m_2024_12_v0.pth"
save_path=$this_file_path/Data/esm_models
mkdir -p $save_path

if [ ! -f "$save_path/$(basename $mode_pt_url)" ]; then
  echo "Downloading ESMC model..."
  wget -P $save_path $mode_pt_url
else
  echo "ESMC model already exists, skipping download."
fi
