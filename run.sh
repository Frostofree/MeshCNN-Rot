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
--superepoch base 50 \
# --superepoch decay 20 min 100 \