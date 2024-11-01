#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:23:10 2024

@author: fuyulei
"""
# 图片合成视频
import cv2
import os
from natsort import natsorted


image_folder = 'figure/' #图片所在文件夹
fps = 5 #帧数
images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
images = natsorted(images)  # 确保图片按顺序排列

first_image_path = os.path.join(image_folder, images[0])
frame = cv2.imread(first_image_path)
height, width, layers = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用适合的编码器
video = cv2.VideoWriter(f'1030-2.mp4', fourcc, fps, (width, height))

#遍历图片并写入视频文件：
for image in images:
    image_path = os.path.join(image_folder, image)
    frame = cv2.imread(image_path)

    # 检查图片尺寸是否匹配
    if frame.shape[1] != width or frame.shape[0] != height:
        print(f"Skipping {image} due to size mismatch.")
        continue

    video.write(frame)  # 写入视频文件   
video.release()
