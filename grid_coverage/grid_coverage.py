#! /usr/bin/python3
# -*- coding:utf-8 -*-

"""

Grid coverage search

author: flztiii

"""

import matplotlib.pyplot as plt
import numpy as np
import math
import copy
from enum import IntEnum
try:
    from grid_map_lib import GridMap
except ImportError:
    raise

# 点
class Point:
    def __init__(self, x, y):
        self.x_ = x
        self.y_ = y
    
    # 重载减法
    def __sub__(self, point):
        return np.hypot(self.x_ - point.x_, self.y_ - point.y_)

# 工具类
class Tools:
    # 找出一个多边形最长边的角点和方向，多边形是闭合的
    @classmethod
    def findPointAndDirection(self, area):
        largest_distance = 0.0
        largest_distance_index = -1
        for i in range(0, len(area) - 1):
            current_point = area[i]
            next_point = area[(i + 1) % len(area)]
            distance = next_point - current_point
            if distance > largest_distance:
                largest_distance_index = i
                largest_distance = distance
        coordinate_center = Point(area[largest_distance_index].x_, area[largest_distance_index].y_)
        coordinate_vector = Point(area[(largest_distance_index + 1) % len(area)].x_ - area[largest_distance_index].x_, area[(largest_distance_index + 1) % len(area)].y_ - area[largest_distance_index].y_)
        return coordinate_center, coordinate_vector
    
    # 拆除点集的x和y
    @classmethod
    def departurePoints(self, points):
        point_x, point_y = [], []
        for point in points:
            point_x.append(point.x_)
            point_y.append(point.y_)
        return point_x, point_y

    # 进行点的坐标转化
    @classmethod
    def transformPoint(self, coordinate_center, theta, point):
        new_x = (point.y_ - coordinate_center.y_) * np.sin(theta) + (point.x_ - coordinate_center.x_) * np.cos(theta)
        new_y = (point.y_ - coordinate_center.y_) * np.cos(theta) - (point.x_ - coordinate_center.x_) * np.sin(theta)
        return Point(new_x, new_y)
    
    # 进行点的还原
    @classmethod
    def restorePoint(self, coordinate_center, theta, point):
        origin_x = point.x_ * np.cos(theta) - point.y_ * np.sin(theta) + coordinate_center.x_
        origin_y = point.x_ * np.sin(theta) + point.y_ * np.cos(theta) + coordinate_center.y_
        return Point(origin_x, origin_y)
    
    # 进行点集坐标转换
    @classmethod
    def transformArea(self, area, coordinate_center, coordinate_vector):
        theta = np.arctan2(coordinate_vector.y_, coordinate_vector.x_)
        new_area = []
        for point in area:
            new_point = self.transformPoint(coordinate_center, theta, point)
            new_area.append(new_point)
        return new_area
    
    # 进行点集坐标还原
    @classmethod 
    def restoreArea(self, area, coordinate_center, coordinate_vector):
        theta = np.arctan2(coordinate_vector.y_, coordinate_vector.x_)
        new_area = []
        for point in area:
            new_point = self.restorePoint(coordinate_center, theta, point)
            new_area.append(new_point)
        return new_area
    
    # 寻找栅格地图的有效边界
    @classmethod
    def searchBoundary(self, grid_map, to_upper=True):
        x_inds = []
        y_ind = None

        # 如果to_upper为true则寻找上边界，如果to_upper为false则寻找下边界
        if to_upper:
            xrange = range(0, grid_map.width_nx_)
            yrange = range(0, grid_map.width_ny_)[::-1]
        else:
            xrange = range(0, grid_map.width_nx_)
            yrange = range(0, grid_map.width_ny_)
        
        for y_index in yrange:
            for x_index in xrange:
                if not grid_map.checkOccupationByIndex(x_index, y_index):
                    y_ind = y_index
                    x_inds.append(x_index)
            if not y_ind is None:
                break
        return x_inds, y_ind

# 行为
class Motion:
    # 扫描行为
    class SweepDirection(IntEnum):
        UP = 1
        DOWN = -1
    # 移动行为
    class MoveDirection(IntEnum):
        LEFT = -1
        RIGHT = 1

# 规划器
class CoveragePlanner:

    def __init__(self, area, resolution, sweep_direction=Motion.SweepDirection.UP, move_direction=Motion.MoveDirection.RIGHT):
        # 基础属性
        self.sweep_direction_ = sweep_direction  # 扫描方向
        self.move_direction_ = move_direction  # 移动方向
        self.turning_windows_ = []  # 移动行为
        self.__updateTurningWindows()  # 更新移动行为
        # 构建代价地图
        self.__buildGridMap(area, resolution)
    
    # 构建代价地图
    def __buildGridMap(self, area, resolution):
        """ 构建栅格地图
            :param area: 探索区域
            :param resolution: 栅格地图分辨率
        """

        # 第一步进行坐标转化
        # 找出区域最长边作为新坐标系的x轴
        self.coordinate_center_, self.coordinate_vector_ = Tools.findPointAndDirection(area)
        # 进行坐标转化
        transformed_area = Tools.transformArea(area, self.coordinate_center_, self.coordinate_vector_)
        # 初始化栅格地图
        self.__initGridMap(transformed_area, resolution)
    
    # 初始化栅格地图
    def __initGridMap(self, polyon, resolution, offset_grid=10):
        # 计算栅格的中心点
        polyon_x, polyon_y = Tools.departurePoints(polyon)
        center_px = np.mean(polyon_x)
        center_py = np.mean(polyon_y)
        # 计算栅格的x轴和y轴宽度
        width_nx = math.ceil((max(polyon_x) - min(polyon_x)) / resolution) + offset_grid
        width_ny = math.ceil((max(polyon_y) - min(polyon_y)) / resolution) + offset_grid
        # 得到栅格地图
        grid_map = GridMap(center_px, center_py, width_nx, width_ny, resolution)
        # 对栅格地图进行赋值
        grid_map.setGridMapbyPolyon(polyon_x, polyon_y, 1.0, False)
        # 进行膨胀
        grid_map.expandGrid()
        # 判断扫描是从下到上还是从上到下
        if self.sweep_direction_ == Motion.SweepDirection.UP:
            goal_nxes, goal_ny = Tools.searchBoundary(grid_map, True)
        else:
            goal_nxes, goal_ny = Tools.searchBoundary(grid_map, False)
        # 记录信息
        self.grid_map_ = grid_map
        self.goal_nxes_ = goal_nxes
        self.goal_ny_ = goal_ny

    # 更新移动行为
    def __updateTurningWindows(self):
        turning_windows = [
            Point(self.move_direction_, 0),
            Point(self.move_direction_, self.sweep_direction_),
            Point(-self.move_direction_, self.sweep_direction_),
            Point(0, self.sweep_direction_),
        ]
        self.turning_windows_ = turning_windows
    
    # 开始规划
    def planning(self):
        # 首先找到起点
        start_nx, start_ny = None, None
        if self.sweep_direction_ == Motion.SweepDirection.UP:
            start_nxes, start_ny = Tools.searchBoundary(self.grid_map_, False)
        else:
            start_nxes, start_ny = Tools.searchBoundary(self.grid_map_, True)
        if self.move_direction_ == Motion.MoveDirection.RIGHT:
            start_nx = min(start_nxes)
        else:
            start_nx = max(start_nxes)
        nstart = Point(start_nx, start_ny)
        # 判断起点是否有效
        if not self.grid_map_.setGridMapValueByIndex(start_nx, start_ny, 0.5):
            print("start is not setable")
            return None
        # 将起始点设置为当前点
        ncurrent = copy.deepcopy(nstart)
        pcurrent = Point(*self.grid_map_.gridMapIndexToPointPos(ncurrent.x_, ncurrent.y_))
        # 生成路径
        planned_path = [pcurrent]
        # 可视化
        # fig, ax = plt.subplots()
        # 从当前点开始移动
        while True:
            # 得到新点
            ncurrent = self.searchNextNode(ncurrent)
            # 判断新点是否存在
            if ncurrent is None:
                print("path cannot find")
                break
            # 计算当前点的坐标
            pcurrent = Point(*self.grid_map_.gridMapIndexToPointPos(ncurrent.x_, ncurrent.y_))
            planned_path.append(pcurrent)
            # 更新栅格地图
            self.grid_map_.setGridMapValueByIndex(ncurrent.x_, ncurrent.y_, 0.5)
            # 可视化
            # self.grid_map_.plotGridMap(ax)
            # plt.pause(1.0)
            # 判断是否到达终点
            if self.__checkFinished():
                print("Done")
                break
        self.grid_map_.plotGridMap()
        # 转化为原始坐标系
        final_path = Tools.restoreArea(planned_path, self.coordinate_center_, self.coordinate_vector_)
        return final_path
    # 寻找下一个点
    def searchNextNode(self, ncurrent):
        # 得到下一个点位置
        nnext = Point(ncurrent.x_ + self.move_direction_, ncurrent.y_)
        # 判断下一个点是否可以
        if not self.grid_map_.checkOccupationByIndex(nnext.x_, nnext.y_, 0.5):
            # 可以，直接返回
            return nnext
        else:
            # 不可以，进一步处理
            # 判断turning_windows中是否存在可以的
            for turning_window in self.turning_windows_:
                # 得到转向后的点
                nnext = Point(ncurrent.x_ + turning_window.x_, ncurrent.y_ + turning_window.y_)
                # 判断是否可以
                if not self.grid_map_.checkOccupationByIndex(nnext.x_, nnext.y_, 0.5):
                    # 可以
                    while not self.grid_map_.checkOccupationByIndex(nnext.x_ + self.move_direction_, nnext.y_, 0.5):
                        nnext = Point(nnext.x_ + self.move_direction_, nnext.y_)
                    self.__swapMotion()
                    return nnext
            # 不可以
            nnext = Point(ncurrent.x_ - self.move_direction_, ncurrent.y_)
            if self.grid_map_.checkOccupationByIndex(nnext.x_, nnext.y_):
                return None
            else:
                return nnext

    # 更新运动
    def __swapMotion(self):
        self.move_direction_ *= -1
        self.__updateTurningWindows()
    
    # 判断是否结束
    def __checkFinished(self):
        for x_inx in self.goal_nxes_:
            if not self.grid_map_.checkOccupationByIndex(x_inx, self.goal_ny_, 0.5):
                return False
        return True

# 主函数
def main():
    # 初始化区域角点
    area = [
        Point(0.0, 0.0),
        Point(20.0, -20.0),
        Point(50.0, 0.0),
        Point(100.0, 30.0),
        Point(130.0, 60.0),
        Point(40.0, 80.0),
        Point(0.0, 0.0),
    ]
    # 初始化分辨率
    resolution = 5.0

    # 构建规划器
    coverage_planner = CoveragePlanner(area, resolution)
    planned_path = coverage_planner.planning()
    px, py = Tools.departurePoints(planned_path)
    ox, oy = Tools.departurePoints(area)
    for ipx, ipy in zip(px, py):
        plt.cla()
        plt.plot(ox, oy, "-xb")
        plt.plot(px, py, "-r")
        plt.plot(ipx, ipy, "or")
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.1)

    plt.cla()
    plt.plot(ox, oy, "-xb")
    plt.plot(px, py, "-r")
    plt.axis("equal")
    plt.grid(True)
    plt.pause(0.1)
    plt.show()

if __name__ == "__main__":
    main()