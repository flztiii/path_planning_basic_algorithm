#! /usr/bin/python3
#-*- coding: utf-8 -*-

"""

Probabilistic roadmap planning
author: flztiii

"""

import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.spatial
import random
import os

# 全局变量
SAMPLE_NUMBER = 200  # 采样点个数
KNN = 10  # 节点邻域个数
MAX_DIS = 30.0  # 邻域最长距离

# 节点类
class Node:
    def __init__(self, index, prev, cost):
        self.index_ = index  # 对应下标
        self.prev_ = prev  # 对应前一个节点下标
        self.cost_ = cost  # 到达此节点花费的代价

# 查找树类
class KDTree:
    def __init__(self, data):
        self.tree_ = scipy.spatial.cKDTree(data)  # 构造kdtree
    
    def search(self, input, k = 1):
        """
            查找输入数据在kdtree中的k个最近邻
            :param input: 输入数据，类型为numpy.array
            :param k: 需要查找的近邻数量
        """
        
        distances, indexes = self.tree_.query(input, k)
        if k == 1:
            distances = [distances]
            indexes = [indexes]
        return indexes, distances

# RoadMap类
class RoadMap:
    def __init__(self, data, nexts):
        self.data_ = data  # 地图中的节点数据
        self.nexts_= nexts  # 每一个节点的邻域节点

    # 通过下标获取节点数据
    def getDataByIndex(self, index):
        return self.data_[index]
    
    # 通过下标获取图中相邻节点
    def getNextsByIndex(self, index):
        return self.nexts_[index]

# 判断是否发生碰撞
def isCollision(current_point, next_point, robot_size, obstacle_search_tree):
    dis = math.sqrt((current_point[0] - next_point[0]) ** 2 + (current_point[1] - next_point[1]) ** 2)
    # 注意math.atan2中的参数不能两个都是反过来，是有方向性的
    # theta = math.atan2(current_point[1] - next_point[1], current_point[0] - next_point[0])是错误的
    theta = math.atan2(next_point[1] - current_point[1], next_point[0] - current_point[0])
    # 如果距离大于最大值，直接判断为碰撞
    if dis > MAX_DIS:
        return True
    # 计算路径上特定点是否与障碍物点距离过小
    nstep = round(dis / robot_size)
    x = current_point[0]
    y = current_point[1]
    for i in range(nstep):
        indexes, distances = obstacle_search_tree.search(np.array([x, y]))
        if distances[0] <= robot_size:
            return True
        x += robot_size * math.cos(theta)
        y += robot_size * math.sin(theta)
    # 计算终点与障碍物距离是否过小
    indexes, distances = obstacle_search_tree.search(next_point)
    if distances[0] <= robot_size:
            return True
    return False

# 判断是否发生碰撞（原版）
def is_collision(sx, sy, gx, gy, rr, okdtree):
    x = sx
    y = sy
    dx = gx - sx
    dy = gy - sy
    yaw = math.atan2(gy - sy, gx - sx)
    d = math.sqrt(dx**2 + dy**2)

    if d >= MAX_DIS:
        return True

    D = rr
    nstep = round(d / D)

    for i in range(nstep):
        idxs, dist = okdtree.search(np.array([x, y]))
        if dist[0] <= rr:
            return True  # collision
        x += D * math.cos(yaw)
        y += D * math.sin(yaw)

    # goal point check
    idxs, dist = okdtree.search(np.array([gx, gy]))
    if dist[0] <= rr:
        return True  # collision

    return False  # OK

# 构建road map
def buildRoadMap(sx, sy, gx, gy, ox, oy, robot_size):
    # 构造障碍物搜索树
    obstacle_search_tree = KDTree(np.vstack((ox, oy)).T)
    # 得到区域边界
    min_x = min(ox)
    max_x = max(ox)
    min_y = min(oy)
    max_y = max(oy)
    width = max_x - min_x
    height = max_y - min_y
    # 生成随机点
    sampled_points = []
    for i in range(SAMPLE_NUMBER):
        # 首先在区域中进行坐标采样
        sample_x = random.random() * width + min_x
        sample_y = random.random() * height + min_y
        # 判断采样点与障碍物的距离，如果满足条件则加入列表
        indexes, distances = obstacle_search_tree.search(np.array([sample_x, sample_y]))
        if distances[0] > robot_size:
            sampled_points.append(np.array([sample_x, sample_y]))
    # 将起点和终点也加入采样点之中
    sampled_points.append(np.array([sx, sy]))
    sampled_points.append(np.array([gx, gy]))
    # 得到随机点后，下一步计算每个点的邻域
    sampled_points_neighbors = []
    sampled_points_tree = KDTree(sampled_points)
    for i in range(len(sampled_points)):
        # 首先计算每个点到这个点的距离，并以距离从小到大进行排序输出
        indexes, distances = sampled_points_tree.search(sampled_points[i], len(sampled_points))
        # 从第二个开始判断是否满足条件
        neighbors = []
        for j in range(1, len(indexes)):
            # 首先判断距离是否大于最大阈值，如果大于直接退出循环
            if distances[j] > MAX_DIS:
                break
            # 如果小于阈值，判断是否与障碍物相交
            current_point = sampled_points[i]
            next_point = sampled_points[indexes[j]]
            if not isCollision(current_point, next_point, robot_size, obstacle_search_tree):
                neighbors.append(indexes[j])
            if len(neighbors) >= KNN:
                break
        sampled_points_neighbors.append(neighbors)
    # 构建road map
    road_map = RoadMap(sampled_points, sampled_points_neighbors)
    return road_map

# 利用Dijkstra算法进行搜索
def dijkstraSearch(road_map, sx, sy, gx, gy):
    # 首先构建起始节点和目标节点
    start_node = Node(len(road_map.data_) - 2, -1, 0.0)
    goal_node = Node(len(road_map.data_) - 1, -1, 0.0)
    # 初始化查询开放集合与查询关闭集合
    open_set, close_set = dict(), dict()
    open_set[len(road_map.data_) - 2] = start_node
    # 开始循环路径搜索
    while True:
        # 从查询开放集合
        if not open_set:
            print("can't find path")
            return None, None
        # 得到开放集合中代价最低的节点
        current_idx = min(open_set, key = lambda o: open_set[o].cost_)
        current_node = open_set[current_idx]
        # 进行可视化
        if len(close_set.keys()) % 2 == 0:
            plt.plot(road_map.getDataByIndex(current_node.index_)[0], road_map.getDataByIndex(current_node.index_)[1], "xg")
            plt.pause(0.001)
        # 判断当前点是否为目标点
        if current_idx == len(road_map.data_) - 1:
            # 当前点就是目标点
            print("find path")
            goal_node = current_node
            break
        # 如果不是目标点，将其从开放集合中删除，加入关闭集合
        del open_set[current_idx]
        close_set[current_idx] = current_node
        # 判断当前点的邻域能否加入查询开放集合
        for idx in road_map.getNextsByIndex(current_idx):
            # 构建下一个节点
            dis = math.sqrt((road_map.getDataByIndex(current_idx)[0] - road_map.getDataByIndex(idx)[0]) ** 2 + (road_map.getDataByIndex(current_idx)[1] - road_map.getDataByIndex(idx)[1]) ** 2)
            next_node = Node(idx, current_idx, current_node.cost_ + dis)
            # 如果当前下标在闭合集合内，不能加入开放集合
            if idx in close_set.keys():
                continue
            # 如果不在闭合也不在开放集合，加入集合内
            if not idx in open_set.keys():
                open_set[idx] = next_node
            else:
                # 如果在放开集合内
                if open_set[idx].cost_ > next_node.cost_:
                    open_set[idx] = next_node

    # 完成路径的查询后，开始构建路径
    path_x, path_y = [gx], [gy]
    check_node = goal_node
    while check_node.prev_ != -1:
        check_node = close_set[check_node.prev_]
        path_x.append(road_map.getDataByIndex(check_node.index_)[0])
        path_y.append(road_map.getDataByIndex(check_node.index_)[1])
    return path_x, path_y

# 进行road map可视化
def plotRoadMap(road_map):
    for i in range(len(road_map.data_)):
        data = road_map.data_[i]
        plt.scatter(data[0], data[1], color = "b", s = 10)

# 进行PRM规划
def PRMPlanning(sx, sy, gx, gy, ox, oy, robot_size):
    # 构造road map
    road_map = buildRoadMap(sx, sy, gx, gy, ox, oy, robot_size)
    # 进行可视化
    plotRoadMap(road_map)
    # 利用Dijkstra算法进行搜索
    path_x, path_y = dijkstraSearch(road_map, sx, sy, gx, gy)
    return path_x, path_y

# 主函数
def main():
    print(os.path.abspath(__file__) + " start!!")

    # 起点和终点的位置
    sx = 10.0  # [m]
    sy = 10.0  # [m]
    gx = 50.0  # [m]
    gy = 50.0  # [m]
    robot_size = 5.0  # [m]

    # 障碍物点位置
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

    # 进行可视化
    plt.plot(sx, sy, '^r')
    plt.plot(gx, gy, '^c')
    plt.plot(ox, oy, '.k')
    plt.axis('equal')
    plt.grid(True)
    
    # 进行PRM规划
    path_x, path_y = PRMPlanning(sx, sy, gx, gy, ox, oy, robot_size)

    if path_x is None:
        return

    # 进行可视化
    plt.plot(path_x, path_y, '-r')
    plt.show()

if __name__ == "__main__":
    main()