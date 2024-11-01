#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:39:49 2024
@author: fuyulei
"""

# 视频分帧
import cv2 as cv
import logging

# log information settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

vc = cv.VideoCapture('Video/20230531_phase/20230531.mp4')

if vc.isOpened():  #判断是否正常打开
    rval, frame = vc.read()
else:
    rval = False

count = 1 # count the number of pictures
frame_interval = 10 # video frame count interval frequency
frame_interval_count = 0

while rval:
    rval, frame = vc.read()
    if (frame_interval_count % frame_interval == 0):
        image_path = 'Video/20230531_phase/pic/{}.jpg'.format(str(count))
        #frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # convert into gray scale images
        cv.imwrite(image_path, frame)
        logging.info('num: ' + str(count) + ', frame: ' + str(frame_interval_count))
        count += 1
    frame_interval_count += 1
    cv.waitKey(1)
vc.release()