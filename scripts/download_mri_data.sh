#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
DATA_DIR=$(realpath "$SCRIPT_DIR/../data")
mkdir -p "$DATA_DIR"
cd "$DATA_DIR" || exit 1
echo "Downloading brain tumor MRI dataset..."
curl -L -o $DATA_DIR/brain-tumor-mri-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/masoudnickparvar/brain-tumor-mri-dataset
echo "Unzipping brain tumor MRI dataset..."
7z x brain-tumor-mri-dataset.zip
echo "Removing zip file..."
rm brain-tumor-mri-dataset.zip