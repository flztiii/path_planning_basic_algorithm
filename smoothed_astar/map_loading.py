#! /usr/bin/python3
#! -*- coding: utf-8 -*-

"""
    map loading tool
    author: flztiii

"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import cv2
import grid_astar

class MapLoader:
    def __init__(self):
        pass

    # 加载地图
    def loadMap(self, path, resolution = 0.05):
        img = cv2.imread(path)
        # 获取图片宽度和高度
        width, height = img.shape[:2][::-1]
        # 转为灰度图
        img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        print(img_gray.shape)
        print(width, height)
        # 读取障碍物栅格
        obstacles = []
        for i in range(0, width):
            for j in range(0, height):
                if img_gray[j, i] < 122:
                    point = grid_astar.Point(i * resolution, j * resolution)
                    obstacles.append(point)
        return obstacles
                

if __name__ == "__main__":
    map_loader = MapLoader()
    path = '/home/flztiii/pythonrobotic_myself/smoothed_astar/data/highwayw.png'
    obstacles = map_loader.loadMap(path)
    print(len(obstacles))
    # 进行可视化
    obstalces_x, obstacles_y = grid_astar.Tools.departPoints(obstacles)
    plt.plot(obstalces_x, obstacles_y, ".k")
    plt.grid(True)
    plt.axis("equal")
    plt.show()
