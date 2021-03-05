#! /usr/bin/python3
# -*- coding: utf-8 -*-

"""

potential field algorithm

author: flztiii

"""

import numpy as np
import matplotlib.pyplot as plt 

# 点
class Point:
    def __init__(self, x, y):
        self.x_ = x
        self.y_ = y

    # 重载等于
    def __eq__(self, point):
        if self.x_ == point.x_ and self.y_ == point.y_:
            return True
        else:
            return False
    # 重载减法
    def __sub__(self, point):
        return np.hypot(self.x_ - point.x_, self.y_ - point.y_)

# 行为
class Motion:
    def __init__(self, mv_x, mv_y):
        self.mv_x_ = mv_x
        self.mv_y_ = mv_y

# 初始设置
class Config:
    kp_ = 5.0  # 引力势能权重参数
    eta_ = 100.0  # 斥力势能权重参数
    area_width_ = 30.0  # 势能场扩展宽度
    resolution_ = 0.5  # 栅格地图分辨率
    robot_size_ = 1.0  # 机器人大小
    detect_radius_ = 5.0  # 障碍物判定范围，大于阈值则不考虑
    motion_ = [
        Motion(1, 0),
        Motion(0, 1),
        Motion(-1, 0),
        Motion(0, -1),
        Motion(1, -1),
        Motion(-1, 1),
        Motion(1, 1),
        Motion(-1, -1)
    ]  # 可以采取的行为

# 工具函数
class Tools:
    # 计算点采取行为后得到的新点
    @classmethod
    def action(self, point, motion):
        return Point(point.x_ + motion.mv_x_, point.y_ + motion.mv_y_)
    
    # 获取点集合的x, y列表
    @classmethod
    def departurePoints(self, points):
        x, y = [], []
        for point in points:
            x.append(point.x_)
            y.append(point.y_)
        return x, y
    
    # 由序号转化为坐标
    @classmethod
    def indexToPos(self, index, min_pos, resolution):
        return index * resolution + min_pos
    
    # 由坐标转化为序号
    @classmethod
    def posToIndex(self, pos, min_pos, resolution):
        return round((pos - min_pos) / resolution)
    
    # 可视化人工势场
    @classmethod
    def drawHeatmap(self, data):
        data = np.array(data).T
        plt.pcolor(data, vmax=100.0, cmap=plt.cm.Blues)

# 人工势场规划器
class PotentialFieldPlanner:
    def __init__(self, p_start, p_goal, p_obstacles):
        self.p_start_ = p_start  # 起点
        self.p_goal_ = p_goal  # 目标点
        self.p_obstacles_ = p_obstacles  # 障碍物点
        self.__buildPotentialField()  # 构建人工势场
    
    # 构建人工势场
    def __buildPotentialField(self):
        # 首先计算区域边界
        self.min_px_ = min(self.p_obstacles_, key=lambda o: o.x_).x_ - Config.area_width_ * 0.5
        self.min_py_ = min(self.p_obstacles_, key=lambda o: o.y_).y_ - Config.area_width_ * 0.5
        self.max_px_ = max(self.p_obstacles_, key=lambda o: o.x_).x_ + Config.area_width_ * 0.5
        self.max_py_ = max(self.p_obstacles_, key=lambda o: o.y_).y_ + Config.area_width_ * 0.5
        print("field area minx:", self.min_px_)
        print("field area miny:", self.min_py_)
        print("field area maxx:", self.max_px_)
        print("field area maxy:", self.max_py_)
        # 计算宽度
        self.width_nx_ = round((self.max_px_ - self.min_px_) / Config.resolution_)
        self.width_ny_ = round((self.max_py_ - self.min_py_) / Config.resolution_)
        print("field nx width:", self.width_nx_)
        print("field ny width:", self.width_ny_)
        # 构建人工势场
        # 初始化
        potential_field = [[0.0 for j in range(0, self.width_ny_)] for i in range(0, self.width_nx_)]
        # 遍历势场中每个栅格
        for ix in range(0, self.width_nx_):
            for iy in range(0, self.width_ny_):
                # 计算当前栅格对应的坐标
                current_point = Point(Tools.indexToPos(ix, self.min_px_, Config.resolution_), Tools.indexToPos(iy, self.min_py_, Config.resolution_))
                # 计算坐标的引力势能
                attractive_potential = self.__calcAttractivePotential(current_point)
                # 计算坐标的斥力势能
                repulsive_potential = self.__calcRepulsivePotential(current_point)
                # 计算总势能
                potential = attractive_potential + repulsive_potential
                potential_field[ix][iy] = potential
        # 可视化人工势场图
        Tools.drawHeatmap(potential_field)
        self.potential_field_ = potential_field
    
    # 计算引力势能
    def __calcAttractivePotential(self, point):
        return 0.5 * Config.kp_ * (point - self.p_goal_)
    
    # 计算斥力势能
    def __calcRepulsivePotential(self, point):
        # 计算离当前点最近的障碍物和距离
        min_distance = float("inf")
        min_index = -1
        for i, _ in enumerate(self.p_obstacles_):
            distance = point - self.p_obstacles_[i]
            if distance < min_distance:
                min_distance = distance
                min_index = i
        # 根据最小距离计算势能
        if min_distance >= Config.detect_radius_:
            # 如果障碍物距离过大，则不考虑
            return 0.0
        elif min_distance <= Config.robot_size_:
            # 如果障碍物距离过小，则势能无限大
            # return float("inf")
            return 10000.0
        else:
            return 0.5 * Config.eta_ * (1.0 / min_distance - 1.0 / Config.detect_radius_) ** 2

    # 利用人工势场进行规划
    def planning(self):
        # 获得起点、目标点对应的栅格
        nstart = Point(Tools.posToIndex(self.p_start_.x_, self.min_px_, Config.resolution_), Tools.posToIndex(self.p_start_.y_, self.min_py_, Config.resolution_))
        ngoal = Point(Tools.posToIndex(self.p_goal_.x_, self.min_px_, Config.resolution_), Tools.posToIndex(self.p_goal_.y_, self.min_py_, Config.resolution_))
        # 可视化起点和终点
        plt.plot(nstart.x_, nstart.y_, "*k")
        plt.plot(ngoal.x_, ngoal.y_, "*m")
        # 将当前点设置为起点
        current_node = nstart
        current_pos = self.p_start_
        # 初始化最终路径
        planned_path = [self.p_start_]
        # 计算当前点到终点的距离
        distance_to_goal = current_pos - self.p_goal_
        # 开始循环梯度下降寻找目标点
        while distance_to_goal > Config.resolution_:
            # 找出当前点以固定行为能够达到最小势能的点
            min_potential = float("inf")
            min_index = -1
            for i, _ in enumerate(Config.motion_):
                # 遍历行为
                motion = Config.motion_[i]
                # 计算采取行为后的新栅格
                next_node = Tools.action(current_node, motion)
                # 判断新栅格的势能是否更小
                if self.potential_field_[next_node.x_][next_node.y_] < min_potential:
                    min_potential = self.potential_field_[next_node.x_][next_node.y_]
                    min_index = i
            # 得到最低势能栅格进行更新
            current_node = Tools.action(current_node, Config.motion_[min_index])
            current_pos = Point(Tools.indexToPos(current_node.x_, self.min_px_, Config.resolution_), Tools.indexToPos(current_node.y_, self.min_py_, Config.resolution_))
            distance_to_goal = current_pos - self.p_goal_
            planned_path.append(current_pos)
            # 可视化
            plt.plot(current_node.x_, current_node.y_, ".r")
            plt.pause(0.01)
        print("Goal Reached")
        return planned_path

# 主函数
def main():
    # 起点和终点
    p_start = Point(0.0, 10.0)
    p_goal = Point(30.0, 30.0)
    # 障碍物点
    p_obstacles = [
        Point(15.0, 25.0),
        Point(5.0, 15.0),
        Point(20.0, 26.0),
        Point(25.0, 25.0)
    ]
    # 网格可视化
    plt.grid(True)
    plt.axis("equal")
    # 进行规划
    potential_field_planner = PotentialFieldPlanner(p_start, p_goal, p_obstacles)
    planned_path = potential_field_planner.planning()
    plt.show()

if __name__ == "__main__":
    print(__file__, "start")
    main()