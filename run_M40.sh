python train.py \
--dataroot datasets/M40_new_new-backed \
--name M40_Rotated_test_new \
--ncf 32 64 128 \
--pool_res 1900 1700 1500 \
--norm group \
--resblocks 1 \
--flip_edges 0.2 \
--slide_verts 0.2 \
--num_aug 0 \
--niter_decay 100 \
--ninput_edges 2000 \
--batch_size  128 \
--save_epoch_freq 1 \
--run_test_freq 1 \
--which_epoch latest \
--num_threads 8 \

# python train.py \
# --dataroot datasets/M40_new-backed \
# --name M40_Rotated_cache_test \
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
# --which_epoch latest \
# # --num_threads 8 \





