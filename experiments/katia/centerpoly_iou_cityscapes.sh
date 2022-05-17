cd src

conda activate CenterPoly

# training from scratch
python main.py polydet --val_intervals 24 --exp_id iou_loss --elliptical_gt --poly_weight 1 --nbr_points 16 --dataset cityscapes --arch smallhourglass  --batch_size 4 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth --poly_loss iou --num_epochs 60

python test.py polydet --exp_id iou_loss_TEST --nbr_points 16 --dataset cityscapes --arch smallhourglass --load_model ../exp/cityscapes/polydet/iou_loss/model_best.pth

#with pretrained CenterPoly
python main.py polydet --val_intervals 24 --exp_id pretrained_iou_loss --elliptical_gt --poly_weight 1 --nbr_points 16 --dataset cityscapes --arch smallhourglass  --batch_size 4 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_16_pw1_B/model_best.pth --poly_loss iou --num_epochs 240

python test.py polydet --exp_id pretrained_iou_loss_TEST --nbr_points 16 --dataset cityscapes --arch smallhourglass --load_model ../exp/cityscapes/polydet/pretrained_iou_loss/model_best.pth
