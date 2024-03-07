#!/usr/bin/env bash

## run the test and export collapses
python test.py \
--dataroot datasets/M40_new_small \
--name M40_Rotated_test \
--ncf 32 64 128 \
--pool_res 1900 1700 1500 \
--norm group \
--resblocks 1 \
--ninput_edges 2000 \
--batch_size 1 \
--which_epoch 10 \
--num_aug 20 \

echo "Done for epoch 5 patch size 8"
