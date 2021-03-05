#! /usr/bin/python3
# -*- coding: utf-8 -*-

"""

Voronoi roadmap planning
author: flztiii

"""

import numpy as np
import math
import scipy.spatial
import matplotlib.pyplot as plt

# 全局变量
KNN = 10  # 邻居点个数
MAX_DISTANCE = 30.0  # 两邻居之间的最长距离

# 节点类
class Node:
    def __init__(self, index, cost, prev_index = -1):
        self.index_ = index
        self.cost_ = cost
        self.prev_index_ = prev_index

# KD树类
class KDTree:
    def __init__(self, data):
        # 输入的data为numpy数组
        self.kd_tree_ = scipy.spatial.KDTree(data)

    # 查找KD树中离给出点最近的k个点
    def searchKNeighbor(self, search_datas, k = 1):
        distances, indexes = [],[]
        for data in search_datas:
            distance, index = self.kd_tree_.query(data, k)
            distances.append(distance)
            indexes.append(index)
        return distances, indexes
    
    # 查找KD树中离给出点距离小于thred的点
    def searchRangeNeighbor(self, search_datas, thred):
        indexes = []
        for data in search_datas:
            index = self.kd_tree_.query_ball_tree(data, thred)
            indexes.append(index)
        return indexes

# 生成采样图
def generateSamplingMap(sx, sy, gx, gy, ox, oy):
    obstacle_points = np.vstack((ox, oy)).T
    # 加入维诺图采样
    voronoi_map = scipy.spatial.Voronoi(obstacle_points)
    sample_x, sample_y = [], []
    for point in voronoi_map.vertices:
        sample_x.append(point[0])
        sample_y.append(point[1])
    # 加入起点和终点
    sample_x.append(sx)
    sample_x.append(gx)
    sample_y.append(sy)
    sample_y.append(gy)
    return sample_x, sample_y

# 判断是否发生碰撞
def isCollision(sx, sy, ex, ey, obstacle_kdtree, robot_size):
    distance = math.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)
    theta = math.atan2(ey - sy, ex - sx)
    steps = math.floor(distance / robot_size)
    if distance > MAX_DISTANCE:
        return True
    for i in range(0, steps):
        x = sx + i * robot_size * math.cos(theta)
        y = sy + i * robot_size * math.sin(theta)
        distances, _ = obstacle_kdtree.searchKNeighbor(np.array([[x, y]]))
        if distances[0] <= robot_size:
            return True
    distances, _ = obstacle_kdtree.searchKNeighbor(np.array([[ex, ey]]))
    if distances[0] <= robot_size:
        return True
    return False
# 生成路图
def generateRoadMap(sample_x, sample_y, ox, oy, robot_size):
    roadmap = []
    # 构建采样点KD树
    sample_map_kdtree= KDTree(np.vstack((sample_x, sample_y)).T)
    # 构建障碍物KD树
    obstacle_kdtree = KDTree(np.vstack((ox, oy)).T)
    sample_length = len(sample_x)
    for (i, x, y) in zip(range(sample_length), sample_x, sample_y):
        # 遍历采样图中的每一个点，找到离其最近的knn个点下标
        _, indexes = sample_map_kdtree.searchKNeighbor(np.array([[x, y]]), KNN)
        indexes = indexes[0]
        # 判断到这knn个点的路径是否与障碍物发生碰撞
        no_collision_indexes = []
        for index in indexes:
            if not isCollision(x, y, sample_x[index], sample_y[index], obstacle_kdtree, robot_size):
                no_collision_indexes.append(index)
        roadmap.append(no_collision_indexes)
    return roadmap

# 进行Dijkstra搜索
def dijkstraSearch(sample_x, sample_y, roadmap):
    # 初始化开放集合和关闭集合
    close_set, open_set = dict(), dict()
    # 初始化起点节点，初始化终点节点
    start_node = Node(len(sample_x) - 2, 0.0)
    end_node = Node(len(sample_x) - 1, 0.0)
    # 将起点加入开放集合
    open_set[len(sample_x) - 2] = start_node
    # 开始进行Dijkstra搜索
    while True:
        # 判断开放集合中是否存在元素
        if not open_set:
            print("can not find a path")
            return None, None
        # 从开放集合中找到代价最小的元素
        current_node_index = min(open_set, key = lambda o: open_set[o].cost_)
        current_node = open_set[current_node_index]
        # 可视化当前搜索点
        if len(close_set.keys()) % 2 == 0:
            plt.plot(sample_x[current_node_index], sample_y[current_node_index], 'xg')
            plt.pause(0.01)
        # 判断当前节点是否为最终节点
        if current_node.index_ == end_node.index_:
            print("find path")
            end_node = current_node
            break
        # 如果不是最终节点，将此节点从开放集合中删除，并加入关闭集合
        del open_set[current_node_index]
        close_set[current_node_index] = current_node
        # 判断当前节点邻居是否可以加入开放集合
        for neighbor_index in roadmap[current_node_index]:
            # 遍历当前节点的邻居
            # 构建邻居节点
            distance = math.sqrt((sample_x[neighbor_index] - sample_x[current_node_index]) ** 2 + (sample_y[neighbor_index] - sample_y[current_node_index]) ** 2)
            neighbor_node = Node(neighbor_index, current_node.cost_ + distance, current_node_index)
            # 判断邻居节点是否在关闭集合中
            if neighbor_index in close_set.keys():
                continue
            # 如果不在关闭集合中
            # 判断邻居节点是否在开放集合中
            if neighbor_index in open_set.keys():
                # 如果在开放集合中
                # 判断其与开放集合中相比代价是否更小
                if open_set[neighbor_index].cost_ > neighbor_node.cost_:
                    # 替代开放集合
                    open_set[neighbor_index] = neighbor_node
            else:
                # 如果不在开放集合中
                # 加入开放集合
                open_set[neighbor_index] = neighbor_node
    # 得到路径后
    current_node = end_node
    path_x = [sample_x[current_node.index_]]
    path_y = [sample_y[current_node.index_]]
    while True:
        current_node = close_set[current_node.prev_index_]
        path_x.append(sample_x[current_node.index_])
        path_y.append(sample_y[current_node.index_])
        if current_node.prev_index_ == -1:
            break
    path_x.reverse()
    path_y.reverse()
    return path_x, path_y


# VRM规划函数
def VRMPlanning(sx, sy, gx, gy, ox, oy, robot_size):
    # 第一步生成采样图
    sample_x, sample_y = generateSamplingMap(sx, sy, gx, gy, ox, oy)    
    # 可视化采样图
    plt.plot(sample_x, sample_y, '.b')

    # 第二步生成路图
    roadmap = generateRoadMap(sample_x, sample_y, ox, oy, robot_size)

    # 第三步进行Dijkstra搜索
    path_x, path_y = dijkstraSearch(sample_x, sample_y, roadmap)

    return path_x, path_y

# 主函数
def main():
    # 初始化起点和终点信息
    sx = 10.0  # [m]
    sy = 10.0  # [m]
    gx = 50.0  # [m]
    gy = 50.0  # [m]

    # 宽度信息
    robot_size = 5.0  # [m]

    # 初始化障碍物信息
    ox = []
    oy = []

    for i in range(60):
        ox.append(i)
        oy.append(0.0)
    for i in range(60):
        ox.append(60.0)
        oy.append(i)
    for i in range(61):
        ox.append(i)
        oy.append(60.0)
    for i in range(61):
        ox.append(0.0)
        oy.append(i)
    for i in range(40):
        ox.append(20.0)
        oy.append(i)
    for i in range(40):
        ox.append(40.0)
        oy.append(60.0 - i)
    
    # 初始化信息可视化
    plt.plot(ox, oy, ".k")
    plt.plot(sx, sy, "^r")
    plt.plot(gx, gy, "^c")
    plt.grid(True)
    plt.axis("equal")

    # 开始进行VRM规划
    path_x, path_y = VRMPlanning(sx, sy, gx, gy, ox, oy, robot_size)
    
    # 可视化最终路径
    plt.plot(path_x, path_y, 'r')
    plt.show()

if __name__ == "__main__":
    main()