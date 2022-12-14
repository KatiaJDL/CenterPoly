conda activate CenterPoly
cd src

# python main.py polydet --val_intervals 24 --exp_id from_ctdet_smhg_1cnv_16_pw1_B_polar --elliptical_gt --poly_weight 1 --nbr_points 16 --dataset cityscapes --arch smallhourglass  --batch_size 4 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth --rep polar
python test.py polydet --exp_id from_ctdet_smhg_1cnv_16_pw1_B_polar_TEST --nbr_points 16 --dataset cityscapes --arch smallhourglass --load_model ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_16_pw1_B_polar/model_best.pth --rep polar

# IoU with pretraining
python main.py polydet --val_intervals 24 --exp_id from_ctdet_smhg_1cnv_16_pw1_B_cospolar --elliptical_gt --poly_weight 1 --nbr_points 16 --dataset cityscapes --arch smallhourglass  --batch_size 4 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth --rep polar
python test.py polydet --exp_id pretrained_iou_loss_polar_TEST --nbr_points 16 --dataset cityscapes --arch smallhourglass --load_model ../exp/cityscapes/polydet/pretrained_iou_loss_polar/model_best.pth --rep polar

#Polar order (modify python file losses.py)
python main.py polydet --val_intervals 24 --exp_id pretrained_iou_loss_polar_order --elliptical_gt --poly_weight 1 --nbr_points 16 --dataset cityscapes --arch smallhourglass  --batch_size 4 --master_batch 4 --lr 2e-4 --load_model ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_16_pw1_B_cospolar/model_best.pth --poly_loss iou --rep polar
python test.py polydet --exp_id pretrained_iou_loss_polar_order_TEST --nbr_points 16 --dataset cityscapes --arch smallhourglass --load_model ../exp/cityscapes/polydet/pretrained_iou_loss_polar_order/model_best.pth --rep polar

#Polar with IoU and fixed angles no pretraining
python main.py polydet --val_intervals 24 --exp_id polar_fixed_iou --elliptical_gt --poly_weight 1 --nbr_points 16 --dataset cityscapes --arch smallhourglass  --batch_size 4 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth --rep polar --poly_loss iou --num_epochs 180
