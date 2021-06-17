from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import json
import numpy as np
import cv2

# io_dic = { '../../cityscapesStuff/BBoxes/train.json':
#             open('../../cityscapesStuff/BBoxes/train.csv', 'r').readlines(),
#            '../../cityscapesStuff/BBoxes/val.json':
#                open('../../cityscapesStuff/BBoxes/val.csv', 'r').readlines(),
#            '../../cityscapesStuff/BBoxes/test.json':
#                open('../../cityscapesStuff/BBoxes/test.csv', 'r').readlines(),
#          }
# io_dic = { '../../cityscapesStuff/BBoxes/train16.json':
#             open('../../cityscapesStuff/BBoxes/train16pts.csv', 'r').readlines(),
#            '../../cityscapesStuff/BBoxes/val16.json':
#                open('../../cityscapesStuff/BBoxes/val16pts.csv', 'r').readlines(),
#          }
# io_dic = { '../../cityscapesStuff/BBoxes/train32.json':
#             open('../../cityscapesStuff/BBoxes/train32pts.csv', 'r').readlines(),
#            '../../cityscapesStuff/BBoxes/val32.json':
#                open('../../cityscapesStuff/BBoxes/val32pts.csv', 'r').readlines(),
#          }
# io_dic = { '../../cityscapesStuff/BBoxes/train256_real_points.json':
#             open('../../cityscapesStuff/BBoxes/train256_real_points.csv', 'r').readlines(),
#            '../../cityscapesStuff/BBoxes/val256_real_points.json':
#                open('../../cityscapesStuff/BBoxes/val256_real_points.csv', 'r').readlines(),
#          }
# io_dic = { '../../cityscapesStuff/BBoxes/train32_real_points.json':
#             open('../../cityscapesStuff/BBoxes/train32_real_points.csv', 'r').readlines(),
#            '../../cityscapesStuff/BBoxes/val32_real_points.json':
#                open('../../cityscapesStuff/BBoxes/val32_real_points.csv', 'r').readlines(),
#          }
# io_dic = { '../../cityscapesStuff/BBoxes/train16_real_points.json':
#             open('../../cityscapesStuff/BBoxes/train16_real_points.csv', 'r').readlines(),
#            '../../cityscapesStuff/BBoxes/val16_real_points.json':
#                open('../../cityscapesStuff/BBoxes/val16_real_points.csv', 'r').readlines(),
#          }
# io_dic = { '../../cityscapesStuff/BBoxes/train8_regular_interval.json':
#             open('../../cityscapesStuff/BBoxes/train8_regular_interval.csv', 'r').readlines(),
#            '../../cityscapesStuff/BBoxes/val8_regular_interval.json':
#                open('../../cityscapesStuff/BBoxes/val8_regular_interval.csv', 'r').readlines(),
#          }
io_dic = { '../../cityscapesStuff/BBoxes/train64_regular_interval.json':
            open('../../cityscapesStuff/BBoxes/train64_regular_interval.csv', 'r').readlines(),
           '../../cityscapesStuff/BBoxes/val64_regular_interval.json':
               open('../../cityscapesStuff/BBoxes/val64_regular_interval.csv', 'r').readlines(),
         }
# io_dic = { '../../cityscapesStuff/BBoxes/train16_regular_interval.json':
#             open('../../cityscapesStuff/BBoxes/train16_regular_interval.csv', 'r').readlines(),
#            '../../cityscapesStuff/BBoxes/val16_regular_interval.json':
#                open('../../cityscapesStuff/BBoxes/val16_regular_interval.csv', 'r').readlines(),
#          }
# io_dic = { '../../KITTIPolyStuff/BBoxes/train16.json':
#             open('../../KITTIPolyStuff/BBoxes/train16.csv', 'r').readlines(),
#            '../../KITTIPolyStuff/BBoxes/val16.json':
#                open('../../KITTIPolyStuff/BBoxes/val16.csv', 'r').readlines(),
#            '../../KITTIPolyStuff/BBoxes/test.json':
#                open('../../KITTIPolyStuff/BBoxes/test.csv', 'r').readlines(),
#            '../../KITTIPolyStuff/BBoxes/trainval16.json':
#                open('../../KITTIPolyStuff/BBoxes/trainval16.csv', 'r').readlines(),
#          }
# io_dic = { '../../KITTIPolyStuff/BBoxes/train32.json':
#             open('../../KITTIPolyStuff/BBoxes/train32.csv', 'r').readlines(),
#            '../../KITTIPolyStuff/BBoxes/val32.json':
#                open('../../KITTIPolyStuff/BBoxes/val32.csv', 'r').readlines(),
#            '../../KITTIPolyStuff/BBoxes/trainval32.json':
#                open('../../KITTIPolyStuff/BBoxes/trainval32.csv', 'r').readlines(),
#          }
# io_dic = { '../../IDDStuff/BBoxes/train16_regular_interval.json':
#             open('../../IDDStuff/BBoxes/train16_regular_interval.csv', 'r').readlines(),
#            '../../IDDStuff/BBoxes/val16_regular_interval.json':
#                open('../../IDDStuff/BBoxes/val16_regular_interval.csv', 'r').readlines(),
#            '../../IDDStuff/BBoxes/test.json':
#                open('../../IDDStuff/BBoxes/test.csv', 'r').readlines(),
#          }


DEBUG = False
import os
# import _init_paths
# from utils.ddd_utils import compute_box_3d, project_to_image, alpha2rot_y
# from utils.ddd_utils import draw_box_3d, unproject_2d_to_3d

'''
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
'''

def _bbox_to_coco_bbox(bbox):
  return [(bbox[0]), (bbox[1]),
          (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]

# cats = ['bus', 'car', 'others', 'van']
cats = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
# cats = ['person', 'rider', 'motorcycle', 'bicycle', 'autorickshaw', 'car', 'truck', 'bus', 'vehicle fallback']
cat_ids = {cat: i + 1 for i, cat in enumerate(cats)}
cat_info = []
for i, cat in enumerate(cats):
  cat_info.append({'name': cat, 'id': i + 1})


for outputfile in io_dic:

    csv_lines = io_dic[outputfile]

    image_to_boxes = {}
    for line in csv_lines:
        items = line.split(',')
        if '1-on-10' in outputfile:
            image_index = int(os.path.basename(items[0].replace('.jpg', '').replace('img', '')))
            if image_index % 10 != 0:
                continue
        BINARY = '_b' in outputfile
        if items[0] in image_to_boxes:
            image_to_boxes[items[0]].append(items[1:])
        else:
            image_to_boxes[items[0]] = [items[1:]]
    ret = {'images': [], 'annotations': [], "categories": cat_info}
    for count, path in enumerate(sorted(image_to_boxes)):

        image_info = {'file_name': path,
                      'id': count,
                      'calib': ''}
        ret['images'].append(image_info)

        for ann_ind, box in enumerate(image_to_boxes[path]):
            x0, y0, x1, y1, label, pseudo_depth = int(float(box[0])), int(float(box[1])), int(float(box[2])), int(float(box[3])), box[4], int(box[5])
            poly_points = [float(item) for item in box[6:]]
            if label.strip() == 'no_object':
                continue

            if not BINARY:
                cat_id = cat_ids[label.strip()]
            else:
                cat_id = 1
            truncated = 0
            occluded = 0
            bbox = [float(x0), float(y0), float(x1), float(y1)]

            ann = {'image_id': count,
                   'id': int(len(ret['annotations']) + 1),
                   'category_id': cat_id,
                   'bbox': _bbox_to_coco_bbox(bbox),
                   'truncated': truncated,
                   'occluded': occluded,
                   'iscrowd': 0,
                   'area': (bbox[3]-bbox[1])*(bbox[2]-bbox[0]),
                   'poly': poly_points,
                   'pseudo_depth': pseudo_depth}
            ret['annotations'].append(ann)

    print("File: ", outputfile)
    print("# images: ", len(ret['images']))
    print("# annotations: ", len(ret['annotations']))
    # import pdb; pdb.set_trace()
    json.dump(ret, open(outputfile, 'w'))
