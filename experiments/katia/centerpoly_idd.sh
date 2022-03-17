#!/usr/bin/env bash

conda activate CenterPoly
cd src

python test.py polydet --nms --exp_id from_cityscapes --nbr_points 16 --dataset IDD --arch smallhourglass  --load_model ../exp/IDD/polydet/from_cityscapes/model_best.pth
