#!/usr/bin/env bash

## run the test and export collapses
python test.py \
--dataroot /media/Data1/siddharth/MCNN-Rot/datasets/M40_heavy \
--name M40_heavy_part_train_30 \
--ncf 32 64 128 \
--pool_res 1900 1700 1500 \
--norm group \
--resblocks 1 \
--ninput_edges 2000 \
--batch_size 1 \
--which_super_epoch 10 \
--num_aug 20 \
--phase test \