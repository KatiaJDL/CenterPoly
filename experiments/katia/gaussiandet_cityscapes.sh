conda activate CenterPoly
cd src

# python main.py gaussiandet --val_intervals 24 --exp_id from_ctdet_d3_pw1_B --elliptical_gt --poly_weight 1 --nbr_points 16 --dataset cityscapes_gaussian --arch smallhourglass  --batch_size 4 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth
python test.py gaussiandet --exp_id from_ctdet_d3_pw1_B_TEST --nbr_points 16 --dataset cityscapes_gaussian --arch smallhourglass --load_model ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_16_pw1_B/model_best.pth

