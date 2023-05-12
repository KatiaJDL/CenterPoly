conda activate CenterPoly
cd src

python main.py polydet --val_intervals 24 --exp_id from_ctdet_smhg_1cnv_16_pw1_iou_l1_loss_polar --elliptical_gt --poly_weight 1 --nbr_points 16 --dataset cityscapes --arch smallhourglass  --batch_size 4 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth --poly_loss l1+iou --rep polar
python test.py polydet --exp_id from_ctdet_smhg_1cnv_16_pw1_iou_l1_loss_polar_TEST --nbr_points 16 --dataset cityscapes --arch smallhourglass --load_model ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_16_pw1_iou_l1_loss_polar/model_best.pth --rep polar

python main.py polydet --val_intervals 24 --exp_id from_ctdet_smhg_1cnv_16_pw1_l1_loss_polar --elliptical_gt --poly_weight 1 --nbr_points 16 --dataset cityscapes --arch smallhourglass  --batch_size 4 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth --poly_loss l1 --rep polar
python test.py polydet --exp_id from_ctdet_smhg_1cnv_16_pw1_l1_loss_polar_TEST --nbr_points 16 --dataset cityscapes --arch smallhourglass --load_model ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_16_pw1_l1_loss_polar/model_best.pth --rep polar

python main.py polydet --val_intervals 24 --exp_id from_ctdet_smhg_1cnv_16_pw1_l1_order_loss_polar --elliptical_gt --poly_weight 1 --nbr_points 16 --dataset cityscapes --arch smallhourglass  --batch_size 4 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth --poly_loss l1 --rep polar --polar_order
python test.py polydet --exp_id from_ctdet_smhg_1cnv_16_pw1_l1_order_loss_polar_TEST --nbr_points 16 --dataset cityscapes --arch smallhourglass --load_model ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_16_pw1_l1_order_loss_polar/model_best.pth --rep polar

python main.py polydet --val_intervals 24 --exp_id from_ctdet_smhg_1cnv_16_pw1_iou_l1_order_loss_polar --elliptical_gt --poly_weight 1 --nbr_points 16 --dataset cityscapes --arch smallhourglass  --batch_size 4 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth --poly_loss l1+iou --rep polar --polar_order
python test.py polydet --exp_id from_ctdet_smhg_1cnv_16_pw1_iou_l1_order_loss_polar_TEST --nbr_points 16 --dataset cityscapes --arch smallhourglass --load_model ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_16_pw1_iou_l1_order_loss_polar/model_best.pth --rep polar
