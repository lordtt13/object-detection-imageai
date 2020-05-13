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
    '--path',
    help = 'Path to Image',
    type = str,
    required = True
)

args = parser.parse_args()
arguments = args.__dict__

detector = ObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath("yolo-tiny.h5?raw=true")
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image = arguments.pop('path'), \
                                             output_image_path = "new.jpg", \
                                                 minimum_percentage_probability = 30)

for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )


cv2.imshow('img', cv2.imread('new.jpg'))    