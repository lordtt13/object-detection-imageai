#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 09:57:31 2020

@author: tanmay
"""

import argparse

from imageai.Detection import VideoObjectDetection


parser = argparse.ArgumentParser()
parser.add_argument(
    '--path',
    help = 'Path to Video',
    type = str,
    required = True
)

args = parser.parse_args()
arguments = args.__dict__

detector = VideoObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath("yolo-tiny.h5")
detector.loadModel()

video_path = detector.detectObjectsFromVideo(input_file_path = arguments.pop("path"),
                                output_file_path = "new.mp4",
                                frames_per_second = 20, log_progress = True)

print(video_path)
