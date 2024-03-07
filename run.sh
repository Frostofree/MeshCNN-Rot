#!/usr/bin/env bash

# run the training
python train.py \
--dataroot datasets/shrec_16 \
--name shrec16_test_try \
--ncf 64 128 256 256 \
--pool_res 600 450 300 180 \
--norm group \
--resblocks 1 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 20 \
--niter_decay 100 \
--fraction_of_data_per_class 0.5 \

# # Changed all from_numpy to tensors, numpy is not loading lmfao
# python train.py \
# --dataroot datasets/M40 \
# --name M40_features \
# --ncf 32 64 128 \
# --pool_res 1900 1700 1500 \
# --norm group \
# --resblocks 1 \
# --flip_edges 0.2 \
# --slide_verts 0.2 \
# --num_aug 20 \
# --niter_decay 100 \
# --ninput_edges 2000 \
# --batch_size  128 \
# --save_epoch_freq 1 \
# --run_test_freq 1 \

# python train.py \
# --dataroot datasets/M40_r \
# --name M40_Rotated_test_classes_new \
# --ncf 32 64 128 \
# --pool_res 1900 1700 1500 \
# --norm group \
# --resblocks 1 \
# --flip_edges 0.2 \
# --slide_verts 0.2 \
# --num_aug 20 \
# --niter_decay 100 \
# --ninput_edges 2000 \
# --batch_size  16 \
# --save_epoch_freq 20 \

# --run_test_freq  \

# ## run the training
# python train.py \
# --dataroot datasets/M40_512_norm_PCA \
# --name M40 \
# --ncf 64 128 256 \
# --pool_res 1800 1600 1500 \
# --norm group \
# --resblocks 1 \
# --flip_edges 0.2 \
# --slide_verts 0.2 \
# --num_aug 20 \
# --niter_decay 100 \
# --ninput_edges 2000 \

