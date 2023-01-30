conda activate CenterPoly
cd src

# python main.py gaussiandet --val_intervals 24 --exp_id from_ctdet_d3_pw1 --elliptical_gt --poly_weight 1 --nbr_points 3 --dataset cityscapes_gaussian --arch smallhourglass  --batch_size 4 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth
python test.py gaussiandet --exp_id from_ctdet_d3_pw1_TEST --nbr_points 3 --dataset cityscapes_gaussian --arch smallhourglass --load_model ../exp/cityscapes_gaussian/gaussiandet/from_ctdet_d3_pw1/model_last.pth

python demo.py gaussiandet --exp_id from_ctdet_d3_pw1_DEMO --nbr_points 3 --dataset cityscapes_gaussian --arch smallhourglass --load_model ../exp/cityscapes_gaussian/gaussiandet/from_ctdet_d3_pw1/model_last.pth --demo /store/datasets/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png

# python main.py gaussiandet --val_intervals 24 --exp_id from_ctdet_d3_pw1_B --elliptical_gt --poly_weight 1 --nbr_points 16 --dataset cityscapes_gaussian --arch smallhourglass  --batch_size 4 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth
python test.py gaussiandet --exp_id from_ctdet_d3_pw1_B_TEST --nbr_points 16 --dataset cityscapes_gaussian --arch smallhourglass --load_model ../exp/cityscapes/polydet/from_ctdet_smhg_1cnv_16_pw1_B/model_best.pth
