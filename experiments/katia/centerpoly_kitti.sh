#!/usr/bin/env bash

conda activate CenterPoly
cd src

python main.py polydet --val_intervals 24 --exp_id from_cityscapes --poly_weight 1 --depth_weight 0 --elliptical_gt --nbr_points 16 --dataset kitti_poly --arch smallhourglass  --batch_size 4 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_16_pw1_B/model_best.pth

python test.py polydet --nms --keep_res --exp_id cityscapes_model --nbr_points 16 --dataset kitti_poly --arch smallhourglass  --load_model ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_16_pw1_B/model_best.pth

