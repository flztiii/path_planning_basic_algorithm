#! /usr/bin/python3
#! -*- coding:utf-8 -*-

"""

Rapid random tree
author: flztiii

"""

import matplotlib.pyplot as plt
import random
import numpy as np

# 全局变量
PATH_RESOLUTION = 0.1  # 路径采样点距离
THRESHOLD = 0.2  # 离终点距离阈值

# 节点
class Node:
    def __init__(self, x, y):
        self.x_ = x  # 节点的x坐标
        self.y_ = y  # 节点的y坐标
        self.path_x_ = []  # 从上一点到本节点的路径x坐标
        self.path_y_ = []  # 从上一点到本节点的路径y坐标
        self.parent_ = None  # 上一个节点

# RRT规划器
class RRTPlanner:
    def __init__(self, start, goal, obstacle_list, rand_area, rand_rate, expand_dis, max_iter):
        self.start_ = start
        self.goal_ = goal
        self.obstacle_list_ = obstacle_list
        self.rand_area_ = rand_area
        self.rand_rate_ = rand_rate
        self.expand_dis_ = expand_dis
        self.max_iter_ = max_iter
    
    # 开始进行规划
    def planning(self):
        # 初始化起始节点
        self.start_node_ = Node(self.start_[0], self.start_[1])
        # 初始化终止节点
        self.goal_node_ = Node(self.goal_[0],self.goal_[1])
        # 初始化生成树
        self.tree_ = [self.start_node_]
        # 开始进行循环搜索
        i = 0
        while i < self.max_iter_:
            # 首先找到采样点
            random_node = self.getRandomNode()
            # 判断生成树中离采样点最近的点
            nearest_node_index = self.getNearestNodeIndexFromTree(random_node)
            nearest_node = self.tree_[nearest_node_index]
            # 从最近节点向采样点方法进行延伸,得到新的节点
            new_node = self.expandNewNode(nearest_node, random_node)
            # 进行可视化
            self.drawGraph(random_node)
            # 判断从最近节点到新节点是否发生障碍物碰撞
            if not self.isCollision(new_node):
                # 没有发生碰撞
                # 将节点加入生成树
                self.tree_.append(new_node)
                # 判断新节点是否离终点距离小于阈值
                if self.calcNodesDistance(new_node, self.goal_node_) < THRESHOLD:
                    # 到达终点
                    # 计算最终路径
                    return self.getFinalPath(new_node)
            i += 1

    # 找到采样点
    def getRandomNode(self):
        rand = random.random()
        if rand > self.rand_rate_:
            # 取终点
            return Node(self.goal_[0],self.goal_[1])
        else:
            # 取随机点
            return Node(random.uniform(self.rand_area_[0], self.rand_area_[1]), random.uniform(self.rand_area_[0], self.rand_area_[1]))
    
    # 获得新的节点
    def expandNewNode(self, init_node, end_node):
        # 计算初始节点到结束节点的距离和朝向
        distance = self.calcNodesDistance(init_node, end_node)
        theta = np.arctan2(end_node.y_ - init_node.y_, end_node.x_ - init_node.x_)
        # 计算从初始节点到结束节点的路径
        distance = min(distance, self.expand_dis_)
        x, y = init_node.x_, init_node.y_
        path_x, path_y = [], []
        for sample in np.arange(0.0, distance + PATH_RESOLUTION, PATH_RESOLUTION):
            x = sample * np.cos(theta) + init_node.x_
            y = sample * np.sin(theta) + init_node.y_
            path_x.append(x)
            path_y.append(y)
        # 构造新的节点
        new_node = Node(x, y)
        new_node.path_x_= path_x[:-1]
        new_node.path_y_ = path_y[:-1]
        new_node.parent_ = init_node
        return new_node

    # 判断节点是否发生碰撞
    def isCollision(self, node):
        # 计算节点路径上每一个点到障碍物距离是否小于阈值
        for ix, iy in zip(node.path_x_, node.path_y_):
            for obs in self.obstacle_list_:
                distance = np.sqrt((ix - obs[0]) ** 2 + (iy - obs[1]) ** 2)
                if distance <= obs[2]:
                    return True
        return False
    
    # 得到最终路径
    def getFinalPath(self, final_node):
        path_x, path_y = [], []
        node = final_node
        while node.parent_ is not None:
            path_x = node.path_x_ + path_x
            path_y = node.path_y_ + path_y
            node = node.parent_
        return path_x, path_y

    # 计算生成树中离给出节点最近的节点下标
    def getNearestNodeIndexFromTree(self, node):
        distances = [self.calcNodesDistance(node, tree_node) for tree_node in self.tree_]
        min_index = distances.index(min(distances))
        return min_index

    # 计算两点之间的距离
    def calcNodesDistance(self, node_1, node_2):
        return np.sqrt((node_1.x_ - node_2.x_) ** 2 + (node_1.y_ - node_2.y_) ** 2)

    # 可视化
    def drawGraph(self, rnd = None):
        # 清空之前可视化
        plt.clf()
        if rnd is not None:
            plt.plot(rnd.x_, rnd.y_, "^k")
        for node in self.tree_:
            if node.parent_:
                plt.plot(node.path_x_, node.path_y_, "-g")

        for (ox, oy, size) in self.obstacle_list_:
            plt.plot(ox, oy, "ok", ms=30 * size)

        plt.plot(self.start_node_.x_, self.start_node_.y_, "xr")
        plt.plot(self.goal_node_.x_, self.goal_node_.y_, "xr")
        plt.axis([self.rand_area_[0], self.rand_area_[1], self.rand_area_[0], self.rand_area_[1]])
        plt.grid(True)
        plt.pause(0.01)

# 主函数
def main():
    # 初始化起点,终点信息
    start = [0.0, 0.0]
    goal = [5.0, 10.0]
    # 初始化障碍物信息
    obstacle_list = [
        (5, 5, 1),
        (3, 6, 2),
        (3, 8, 2),
        (3, 10, 2),
        (7, 5, 2),
        (9, 5, 2)
    ]  # [x,y,size]
    # 初始化采样
    rand_area=[-2.0, 15.0]
    # 初始化步长
    expand_dis = 1.0
    # 初始化最大迭代次数
    max_iter = 1000
    # 初始化随机点采样概率
    rand_rate = 0.5

    # 开始规划
    rrt_planner = RRTPlanner(start, goal, obstacle_list, rand_area, rand_rate, expand_dis, max_iter)
    path_x, path_y = rrt_planner.planning()

    # 进行可视化
    rrt_planner.drawGraph()
    plt.plot(path_x, path_y)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()