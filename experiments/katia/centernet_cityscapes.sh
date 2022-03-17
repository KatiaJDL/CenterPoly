conda activate CenterPoly
cd src

# python main.py ctdet --val_intervals 2 --exp_id ctdet_1 --dataset cityscapes --arch smallhourglass  --batch_size 4 --master_batch 4 --lr 2e-4 --load_model ../models/ctdet_coco_hg.pth
# python test.py --nms ctdet --exp_id ctdet --dataset cityscapes --arch smallhourglass --load_model ../exp/cityscapes/ctdet/ctdet/model_best.pth
