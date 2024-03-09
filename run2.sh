#!/usr/bin/env bash

#!/usr/bin/env bash

## run the training
# python train.py \
# --dataroot datasets/shrec_16 \
# --name shrec16_test \
# --ncf 64 128 256 256 \
# --pool_res 600 450 300 180 \
# --norm group \
# --resblocks 1 \
# --flip_edges 0.2 \
# --slide_verts 0.2 \
# --num_aug 20 \
# --niter_decay 100 \

# Changed all from_numpy to tensors, numpy is not loading lmfao
python train.py \
--dataroot datasets/M40_small_new \
--name M40_Rotated_test_new \
--ncf 32 64 128 \
--pool_res 1900 1700 1500 \
--norm group \
--resblocks 1 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 20 \
--niter_decay 100 \
--ninput_edges 2000 \
--batch_size  16 \
--run_test_freq 300 \

# python train.py \
# --dataroot datasets/M40_r \
# --name M40_Rotated_test_classes \
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
# --run_test_freq 300 \
# --gpu_ids -1 \

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

