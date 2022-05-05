from ctypes import py_object
import os
import glob
import json
import csv
import numpy as np
import math
from PIL import Image
import cv2
from PIL import Image, ImageDraw

NBR_POINTS = 32

have_instances = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
id_to_label = {24:'person', 25:'rider', 26:'car', 27:'truck', 28:'bus',29:'noinstance', 30:'noinstance', 31:'train', 32:'motorcycle', 33:'bicycle'}

for filename in sorted(glob.glob('../BBoxes/*regular_interval.json', recursive=True)):
    new_filename = filename.replace('.json', '_polar.json')
    data = json.load(open(filename))

    objects = data['annotations']

    for object in objects:
        center_x = object['bbox'][0]
        center_y = object['bbox'][1]
        cartesian = object['poly']

        polar = []
        for i in range(0,len(cartesian), 2):
            x = cartesian[i]-center_x
            y = cartesian[i+1]-center_y

            r = math.sqrt(x*x+y*y)
            theta = math.atan(y/(x+1e-8))
            if x < 0 : 
                theta = theta + math.pi
            polar.append(r)
            polar.append(theta)

        object['poly'] = polar

    #data['annotations'] = objects
    with open(new_filename, 'w') as f:
        f.write(json.dumps(data, sort_keys=True))


#Faire un essai jouet sur un exemple de polygone des images