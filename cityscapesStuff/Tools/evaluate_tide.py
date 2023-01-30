import json

from tidecv import TIDE
import tidecv.datasets as datasets

masks_file ='/store/travail/kajoda/CenterPoly/CenterPoly/exp/cityscapes/ctdet/ctdet/results.json'

f = open(masks_file)

data = json.load(f)

gt = datasets.Cityscapes('/store/datasets/cityscapes/gtFine/train')

#gt = datasets.COCO()

mask_results = datasets.COCOResult(masks_file)

tide = TIDE()

tide.evaluate_range(gt, mask_results, mode=TIDE.BOX)

tide.summarize()

tide.plot()

# pb de division par zéro dans getAP, à investiguer