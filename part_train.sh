#!/usr/bin/env bash

# run the training
python part_train.py \
--dataroot /media/Data1/siddharth/MCNN-Rot/datasets/M40_heavy \
--name M40_heavy_part_train \
--ncf 32 64 128 \
--pool_res 1900 1700 1500 \
--ninput_edges 2000 \
--norm group \
--resblocks 1 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 20 \
--super_epoch 100 \
--superepoch_base 50 \
--part_size 1000 \
--save_epoch_freq 49 \