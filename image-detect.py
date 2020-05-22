#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 08:51:29 2020

@author: tanmay
"""

import cv2
import argparse

from imageai.Detection import ObjectDetection


parser = argparse.ArgumentParser()
parser.add_argument(
    '--load-path',
    help = 'Path to Image',
    type = str,
    required = True
)

parser.add_argument(
    '--model',
    help = 'Model which you want',
    type = str,
    default = "YOLO-tiny",
    options = ["Retinanet", "YOLO", "YOLO-tiny"]
)

parser.add_argument(
    '--save-path',
    help = 'Path to Image',
    type = str,
    default = "new.jpg"
)

args = parser.parse_args()
arguments = args.__dict__

detector = ObjectDetection()

if arguments.pop('model') == "YOLO":
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath("yolo.h5")
elif arguments.pop('model') == "Retinanet":
    detector.setModelTypeAsRetinanet()
    detector.setModelPath("resnet50_coco_best_v2.0.1.h5")
else:
    detector.setModelTypeAsTinyYOLOv3()
    detector.setModelPath("yolo-tiny.h5")

detector.loadModel()
detections = detector.detectObjectsFromImage(input_image = arguments.pop('load-path'), \
                                             output_image_path = arguments.pop('save-path'), \
                                                 minimum_percentage_probability = 30)

for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
