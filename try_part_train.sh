#!/usr/bin/env bash

# run the training
python part_train.py \
--dataroot datasets/M40_sample \
--name M40_sample_part_train_try \
--ncf 32 64 128 \
--pool_res 1900 1700 1500 \
--ninput_edges 2000 \
--norm group \
--resblocks 1 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 20 \
--super_epoch 100 \
--superepoch_base 10 \
--part_size 10 \
--save_epoch_freq 49 \
--continue_part_train \
--which_super_epoch 1 \ 