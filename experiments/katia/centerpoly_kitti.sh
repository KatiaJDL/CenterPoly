#!/usr/bin/env bash

conda activate CenterPoly
cd src

python test.py polydet --nms --keep_res --exp_id cityscapes_model --nbr_points 16 --dataset kitti_poly --arch smallhourglass  --load_model ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_16_pw1_B/model_best.pth

