conda activate CenterPoly
cd src

python main.py diskdet --val_intervals 24 --exp_id test --elliptical_gt --poly_weight 1 --nbr_points 16 --dataset cityscapes --arch smallhourglass  --batch_size 4 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth
