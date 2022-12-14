#!/usr/bin/env bash

conda activate CenterPoly
cd src

#python main.py polydet --val_intervals 1 --exp_id from_smallhourglass --poly_weight 1 --depth_weight 0.1 --elliptical_gt --nbr_points 16 --dataset IDD --arch smallhourglass  --batch_size 4 --master_batch 4 --lr 2e-4 --load_model ../models/Small_Hourglass.pth
python test.py polydet --nms --exp_id from_cityscapes_TEST --nbr_points 16 --dataset IDD --arch smallhourglass  --load_model ../exp/IDD/polydet/from_cityscapes/model_best.pth

# training from scratch with l1 + iou
python main.py polydet --val_intervals 24 --exp_id iou_l1_loss --elliptical_gt --poly_weight 1 --nbr_points 16 --dataset cityscapes --arch smallhourglass  --batch_size 4 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/polydet/iou_l1_loss/model_best.pth --poly_loss l1+iou
python test.py polydet --nms --exp_id iou_l1_loss_TEST --nbr_points 16 --dataset IDD --arch smallhourglass  --load_model ../exp/IDD/polydet/iou_l1_loss/model_best.pth