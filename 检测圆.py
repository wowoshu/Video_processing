# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt

#定义一些常数
#circle_info_list = np.zeros((1000, 2, 3))  #存储信息



def detect_circles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=10, maxRadius=100)
    if circles is None:
        raise ValueError(f"图像 {image_path} 中未检测到任何圆形物体")
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
    return circles

def calculate_speed(prev_circles, curr_circles, time_interval):
    speeds = []
    for (x1, y1, _), (x2, y2, _) in zip(prev_circles, curr_circles):
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        speed = distance / time_interval
        speeds.append(speed)
    return speeds

def match_circles(prev_circles, curr_circles):
    matched_circles = []
    used_indices = set()
    for (x1, y1, r1) in prev_circles:
        min_distance = float('inf')
        matched_circle = None
        matched_index = -1
        for i, (x2, y2, r2) in enumerate(curr_circles):
            if i in used_indices:
                continue
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if distance < min_distance:
                min_distance = distance
                matched_circle = (x2, y2, r2)
                matched_index = i
        used_indices.add(matched_index)
        matched_circles.append(matched_circle)
    return matched_circles

def save_to_csv(circles, speeds, csv_path, frame_number):
    fieldnames = ['Frame', 'ID', 'Center_X', 'Center_Y', 'Radius', 'Speed']
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if frame_number == 1:
            writer.writeheader()
        for i, ((x, y, r), speed) in enumerate(zip(circles, speeds), start=1):
            writer.writerow({'Frame': frame_number, 'ID': i, 'Center_X': x, 'Center_Y': y, 'Radius': r, 'Speed': speed})

def main(image_paths, csv_path, time_interval):
    #前一帧的圆坐标
    prev_circles = None

    for frame_number, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(image_path)
        circles = detect_circles(image)


            # i = 0
            # for (x, y, r) in circles:
            #     circle_info_list[j][i][0] = x
            #     circle_info_list[j][i][1] = y
            #     circle_info_list[j][i][2] = r
            #     i = i + 1    

                
        #按照x坐标排序圆
        sorted_circles = sorted(circles, key=lambda x: x[0])
        #创建一个初始值为零，dim=圆的个数的数组
        speeds = [0] * len(sorted_circles)
        
        if prev_circles is not None:
            matched_circles = match_circles(prev_circles, sorted_circles)
            speeds = calculate_speed(prev_circles, matched_circles, time_interval)
            sorted_circles = matched_circles
        #将匹配上一帧顺序的圆按顺序写入csv文件
        save_to_csv(sorted_circles, speeds, csv_path, frame_number)
        #将本帧圆的信息作为下一帧的基准
        prev_circles = sorted_circles
        #标注圆心、圆轮廓和圆的序号
        for i, (x, y, r) in enumerate(sorted_circles, start=1):
            # 绘制圆心
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            # 绘制圆轮廓
            cv2.circle(image, (x, y), r, (0, 0, 255), 2)
            # 标注序号
            cv2.putText(image, f'{i}', (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        output_path = f'pic_process/annotated_frame_{frame_number}.jpg'
        cv2.imwrite(output_path, image)
        #cv2.imshow('Annotated Circles', image)
        #cv2.waitKey(500)  # 显示每帧500毫秒
    cv2.destroyAllWindows()

# 实际运行
image_paths = []
for i in range(1,428):
    image_paths.append(f'pic/{i}.jpg')
csv_path = 'circles_tracking.csv'
time_interval = 0.03  # 两帧之间的时间间隔，单位是秒
main(image_paths, csv_path, time_interval)


#找一帧图片看看
image = cv2.imread('Video/20230531_phase/176.jpg')
plt.imshow(image)




# 打开视频文件
cap = cv2.VideoCapture('Video/20230531_phase/20230531.mp4')

# 定义帧计数器
frame_count = 0
skip_frames = 60 #每隔几帧读一次


# 读取视频帧
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        # 每一帧的处理逻辑
        if frame_count % skip_frames == 0:
            # 将图片转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #自适应阈值将图片二值化
            adaptive_thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
            plt.imshow(adaptive_thresh)
            #开运算
            kernel = np.ones((3,3),np.uint8)
            opened = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
            # 使用霍夫圆检测找到圆
            circles = cv2.HoughCircles(opened, cv2.HOUGH_GRADIENT, dp=1.2, minDist=8, param1=100, param2=10, minRadius=5, maxRadius=10)
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")

                for (x, y, r) in circles:
                    # 绘制圆心
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    # 绘制圆轮廓
                    #cv2.circle(frame, (x, y), r, (0, 0, 255), 2)
        
                # 保存帧到文件中
                cv2.imwrite(f'Video/20230531_phase/pic/frame_{frame_count}.jpg', frame)

        # 增加帧计数器
        frame_count += 1
        
        
        # 按下 'q' 键退出循环
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
cv2.waitKey(0)
cv2.destroyAllWindows()    


#保存三维数组到不同csv文件 
for i in range(circle_info_list.shape[0]):
    np.savetxt(f'0522pos/slice_{i}.csv',circle_info_list[i],delimiter = ',' )
 

#检测一堆圆
image = cv2.imread('Video/20230531_phase/227.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#自适应阈值将图片二值化
adaptive_thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,15,7)
adaptive_rgb = cv2.cvtColor(adaptive_thresh,cv2.COLOR_BGR2RGB)
plt.imshow(adaptive_rgb)
#开运算
kernel = np.ones((2,2),np.uint8)
opened = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
opened_rgb = cv2.cvtColor(opened,cv2.COLOR_BGR2RGB)
plt.imshow(opened_rgb)

# 使用霍夫圆检测找到圆
circles = cv2.HoughCircles(opened, cv2.HOUGH_GRADIENT, dp=1.2, minDist=8, param1=100, param2=10, minRadius=5, maxRadius=10)

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")

    for (x, y, r) in circles:
        # 绘制圆心
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
        # 绘制圆轮廓
        #cv2.circle(frame, (x, y), r, (0, 0, 255), 2)

    # 保存帧到文件中
    cv2.imwrite(f'Video/20230531_phase/pic/227-1.jpg', image)

