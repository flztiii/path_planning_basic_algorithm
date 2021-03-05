#! /usr/bin/python3
#! -*- coding: utf-8 -*-

"""

Rapid random tree star
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
        self.cost_ = 0.0  # 代价

# RRT规划器
class RRTStarPlanner:
    def __init__(self, start, goal, obstacle_list, rand_area, rand_rate, expand_dis, connect_circle_dist, max_iter):
        self.start_ = start
        self.goal_ = goal
        self.obstacle_list_ = obstacle_list
        self.rand_area_ = rand_area
        self.rand_rate_ = rand_rate
        self.expand_dis_ = expand_dis
        self.max_iter_ = max_iter
        self.connect_circle_dist_ = connect_circle_dist
    
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
            print("Iter:", i, ", number of nodes:", len(self.tree_))
            # 首先找到采样点
            random_node = self.getRandomNode()
            # 判断生成树中离采样点最近的点
            nearest_node_index = self.getNearestNodeIndexFromTree(random_node)
            nearest_node = self.tree_[nearest_node_index]
            # 从最近节点向采样点方法进行延伸,得到新的节点
            new_node = self.expandNewNode(nearest_node, random_node, self.expand_dis_)
            # 找到新节点的在生成树中的邻居节点
            neighbor_node_indexes = self.findNeighborNodeIndexes(new_node)
            # 为新节点更新父节点
            new_node = self.updateParentForNewNode(new_node, neighbor_node_indexes)
            # 进行可视化
            if i % 10 == 0:
                self.drawGraph(random_node)
            # 判断新节点是否有效
            if new_node is not None:
                # 将新节点加入生成树
                self.tree_.append(new_node)
                # 新节点有效,更新邻居节点的连接
                self.updateWire(new_node, neighbor_node_indexes)
            i += 1
        # 遍历完成,开始得到最终路径
        last_index = self.findFinalNode()
        if last_index is None:
            print('no path find')
            exit(0)
        else:
            return self.getFinalPath(self.tree_[last_index])

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
    def expandNewNode(self, init_node, end_node, max_distance=float('inf')):
        # 计算初始节点到结束节点的距离和朝向
        distance = self.calcNodesDistance(init_node, end_node)
        theta = np.arctan2(end_node.y_ - init_node.y_, end_node.x_ - init_node.x_)
        # 计算从初始节点到结束节点的路径
        distance = min(distance, max_distance)
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
        new_node.cost_ = self.calcCost(init_node, new_node)
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
    
    # 计算节点的代价
    def calcCost(self, init_node, end_node):
        return init_node.cost_ + self.calcNodesDistance(init_node, end_node)

    # 寻找邻居节点
    def findNeighborNodeIndexes(self, node):
        # 得到搜索半径
        radius = self.connect_circle_dist_ * np.sqrt(np.log(len(self.tree_) + 1) / (len(self.tree_) + 1))
        indexes = []
        for i, tree_node in enumerate(self.tree_):
            distance = self.calcNodesDistance(node, tree_node)
            if distance <= radius:
                indexes.append(i)
        return indexes
    
    # 为新节点更新父节点
    def updateParentForNewNode(self, node, neighbor_node_indexes):
        # 遍历邻居节点
        valid_temp_node = []
        for neighbor_node_index in neighbor_node_indexes:
            neighbor_node = self.tree_[neighbor_node_index]
            # 构建临时节
            temp_node = self.expandNewNode(neighbor_node, node)
            # 判断临时节点是否发生碰撞
            if not self.isCollision(temp_node):
                # 如果没有发生碰撞,加入列表中
                valid_temp_node.append(temp_node)
        if len(valid_temp_node) == 0:
            # 没有有效节点
            return None
        else:
            # 返回代价最小的
            min_cost_node = min(valid_temp_node, key = lambda temp_node: temp_node.cost_)
            return min_cost_node

    # 更新新的连接关系
    def updateWire(self, node, neighbor_node_indexes):
        # 遍历邻居节点
        for neighbor_node_index in neighbor_node_indexes:
            neighbor_node = self.tree_[neighbor_node_index]
            # 构建临时节点
            temp_node = self.expandNewNode(node, neighbor_node)
            # 判断是否发生碰撞
            if self.isCollision(temp_node):
                # 如果发生碰撞,跳过
                continue
            # 如果不发生碰撞,判断代价
            if temp_node.cost_ < neighbor_node.cost_:
                # 如果新的节点代价更低,更新之前的邻居节点为新的
                self.tree_[neighbor_node_index] = temp_node
                # 更新该邻居的全部子节点
                self.propegateCost(neighbor_node)
    
    # 向子节点传播损失
    def propegateCost(self, node):
        # 遍历全部的生成树节点
        for i, tree_node in enumerate(self.tree_):
            if tree_node.parent_ == node:
                # 找到了子节点,进行损失更新
                self.tree_[i].cost_ = self.calcCost(node, self.tree_[i])
                self.propegateCost(self.tree_[i])

    # 寻找离终点最近的节点
    def findFinalNode(self):
        final_indexes = []
        for i, tree_node in enumerate(self.tree_):
            distance = self.calcNodesDistance(tree_node, self.goal_node_)
            if distance < THRESHOLD:
                final_indexes.append(i)
        # 判断是否找到终点
        if len(final_indexes) == 0:
            # 没有找到终点
            return None
        final_index = min(final_indexes, key = lambda index: self.tree_[index].cost_)
        return final_index

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
    max_iter = 500
    # 初始化随机点采样概率
    rand_rate = 0.5
    # 初始邻居半径
    connect_circle_dist = 50.0

    # 开始规划
    rrt_star_planner = RRTStarPlanner(start, goal, obstacle_list, rand_area, rand_rate, expand_dis, connect_circle_dist, max_iter)
    path_x, path_y = rrt_star_planner.planning()

    # 进行可视化
    rrt_star_planner.drawGraph()
    plt.plot(path_x, path_y, 'r')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()