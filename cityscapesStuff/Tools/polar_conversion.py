import os
import glob
import json
import csv
import numpy as np
import math
import bresenham
from PIL import Image, ImageDraw

METHODS = ['grid_based']  #   # , 'real_points'
COARSE = False
NBR_POINTSS = [24, 32, 40] # 16, 32, 64
# from cityscapes scripts, thee labels have instances:
have_instances = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'pole', 'traffic sign', 'traffic light']
instance_frequencies = {'person':0, 'rider':0, 'car':0, 'truck':0, 'bus':0, 'train':0, 'motorcycle':0, 'bicycle':0, 'pole':0, 'traffic sign':0, 'traffic light':0}

max_objects_per_img = 0
max_point_per_polygon = 0
if COARSE:
    sets = 'train', 'val'
else:
    sets = ['val', 'train']  #, 'train  # , 'test'

for method in METHODS:
    for NBR_POINTS in NBR_POINTSS:
        for data_set in sets:
            if COARSE:
                spamwriter = csv.writer(open('../BBoxes/' + data_set + '_coarse.csv', 'w'), delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
            elif 'test' in data_set:
                spamwriter = csv.writer(open('../BBoxes/' + data_set + '.csv', 'w'), delimiter=',',quotechar='', quoting=csv.QUOTE_NONE)
            else:
                spamwriter = csv.writer(open('../BBoxes/' + data_set + str(NBR_POINTS) + '_' + method +'.csv', 'w'), delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)
            for filename in sorted(glob.glob('/store/datasets/cityscapes/leftImg8bit/' + data_set + '/*/*.png', recursive=True)):
                if COARSE:
                    gt_path = filename.replace('leftImg8bit', 'gtCoarse').replace('.png', '_polygons.json')
                else:
                    gt_path = filename.replace('leftImg8bit', 'gtFine').replace('.png', '_polygons.json')
                # img_path = filename.replace('gtFine', 'leftImg8bit').replace('json', 'png').replace('_polygons', '')
                data = json.load(open(gt_path))
                objects = data['objects']
                objects.reverse()
                count = 0
                for object in objects:
                    label = object['label']
                    if label in have_instances:
                        instance_frequencies[label] += 1
                        bbox = x0, y0, x1, y1 = polygon_to_box(object['polygon'])
                        items = [os.path.abspath(filename), x0, y0, x1, y1, label, count]

                        if 'real_points' in method:

                            while len(object['polygon']) > NBR_POINTS:
                                distances = []
                                for i in range(1, len(object['polygon'])):
                                    distances.append(get_distance(object['polygon'][i - 1], object['polygon'][i]))
                                min_index = np.argsort(distances)[0]
                                del object['polygon'][min_index]

                            while len(object['polygon']) < NBR_POINTS:
                                distances = []
                                for i in range(1, len(object['polygon'])):
                                    distances.append(get_distance(object['polygon'][i-1], object['polygon'][i]))
                                max_index = np.argsort(distances)[-1]
                                new_point = get_mid_point(object['polygon'][max_index], object['polygon'][max_index+1])
                                object['polygon'].insert(max_index+1, new_point)

                            object['polygon'] = rotate_points(object['polygon'], bbox)
                        elif 'grid_based' in method:
                            poly_img = Image.new('L', (2048, 1024), 0)
                            ImageDraw.Draw(poly_img).polygon([tuple(item) for item in object['polygon']], outline=0, fill=255)
                            poly_img = np.array(poly_img)
                            lines = find_grid_lines_from_box(box=bbox, n_points=NBR_POINTS)
                            points_on_border = []
                            for line in lines:
                                ((x0, y0), (x1, y1)) = line
                                points = bresenham.bresenham(x0, y0, x1, y1)
                                points_on_border.append(find_first_non_zero_pixel(points, poly_img))
                            object['polygon'] = points_on_border
                        elif 'regular_interval' in method:
                            poly_img = Image.new('L', (2048, 1024), 0)
                            ImageDraw.Draw(poly_img).polygon([tuple(item) for item in object['polygon']], outline=0, fill=255)
                            poly_img = np.array(poly_img)
                            points_on_box = find_points_from_box(box=bbox, n_points=NBR_POINTS)
                            points_on_border = []
                            ct = int(x0 + ((x1-x0)/2)), int(y0 + ((y1-y0)/2))
                            for point_on_box in points_on_box:
                              line = bresenham.bresenham(int(point_on_box[0]), int(point_on_box[1]), int(ct[0]), int(ct[1]))
                              points_on_border.append(find_first_non_zero_pixel(line, poly_img))
                            del poly_img
                            object['polygon'] = points_on_border

                        elif method == 'on_border':
                            points = set(object['polygon'])
                            for i in range(1, len(object['polygon'])):
                                for point in bresenham.bresenham(int(object['polygon'][i-1][0]),
                                                                 int(object['polygon'][i-1][1]),
                                                                 int(object['polygon'][i][0]),
                                                                 int(object['polygon'][i][1])):
                                    points.add(point)


                        if len(object['polygon']) > max_point_per_polygon:
                            max_point_per_polygon = len(object['polygon'])
                        for point in np.array(object['polygon']).flatten():
                            items.append(point)
                        spamwriter.writerow(tuple(items))
                        count += 1
                if count > max_objects_per_img:
                    max_objects_per_img = count
                if count == 0:
                    spamwriter.writerow((os.path.abspath(filename), -1, -1, -1, -1, 'no_object', 0))
                if data_set == 'test':
                    spamwriter.writerow((os.path.abspath(filename), 0, 0, 1, 1, 'car', 0))

        print('max objects: ', max_objects_per_img)
        print('max nbr points: ', max_point_per_polygon)
print(instance_frequencies)
