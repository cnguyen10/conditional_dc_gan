#!/bin/sh
DEVICE_ID=1  # which GPU is going to be used
export CUDA_VISIBLE_DEVICES=${DEVICE_ID}

BASEDIR=$(cd $(dirname $0) && pwd)

function clean_up() {
    echo "Clean up"
}

# DATASET_PATH="/sda2/datasets/tiny-imagenet-200/train.json"
DATASET_PATH="./json/img_align_celeba.json"

python3 -W ignore "dc_gan.py" \
    --experiment-name "DC-GAN" \
    --run-description "Vanilla DC-GAN" \
    --train-groundtruth-file $DATASET_PATH \
    --num-classes "200" \
    --ngf "64" \
    --ndf "64" \
    --nz "100" \
    --lr "2e-4" \
    --batch-size "128" \
    --num-epochs "100" \
    --jax-platform "cuda"