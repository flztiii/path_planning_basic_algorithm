#! /usr/bin/python3
# -*- coding: utf-8 -*-

"""

grid dijkstra

author: flztiii

"""

import matplotlib.pyplot as plt
import numpy
import math
import copy

# 点
class Point:
    def __init__(self, x, y):
        self.x_ = x  # x轴坐标
        self.y_ = y  # y轴坐标
    
    # 重载减法操作
    def __sub__(self, point):
        return math.sqrt((self.x_ - point.x_) ** 2 + (self.y_ - point.y_) ** 2)

# 节点
class Node:
    def __init__(self, nx, ny, cost, prev_id):
        self.nx_ = nx  # x轴下标
        self.ny_ = ny  # y轴下标
        self.cost_ = cost  # 代价
        self.prev_id_ = prev_id  # 前一个节点id

    # 重置等于操作
    def __eq__(self, node):
        if self.nx_ == node.nx_ and self.ny_ == node.ny_:
            return True
        else:
            return False
    
    # 重置不等于操作
    def __ne__(self, node):
        if self.nx_ == node.nx_ and self.ny_ == node.ny_:
            return False
        else:
            return True

# 行为
class Motion:
    def __init__(self, x_mv, y_mv):
        self.x_mv_ = x_mv  # x轴移动
        self.y_mv_ = y_mv  # y轴移动
        self.cost_ = math.sqrt(x_mv**2 + y_mv**2)  # 移动所需代价

# 设定
class Config:
    def __init__(self):
        self.resolution_ = 2.0  # 栅格分辨率
        self.robot_size_ = 1.0  # 碰撞判断范围
        self.motion_ = [
            Motion(0, 1), 
            Motion(1, 0),
            Motion(1, 1),
            Motion(0, -1),
            Motion(-1, 0),
            Motion(-1, -1), 
            Motion(1, -1), 
            Motion(-1, 1)
        ]  # 可以采取的运动行为

# 工具类
class Tools:
    def __init__(self):
        return
    
    # 将点集的x坐标和y坐标分离
    @classmethod
    def departPoints(self, points):
        x = []
        y = []
        for point in points:
            x.append(point.x_)
            y.append(point.y_)
        return x, y
    
    # 将坐标转化为索引
    @classmethod
    def posToIndex(self, pos, min_pos, resolution):
        return round((pos - min_pos) / resolution)

    # 将索引转化为坐标
    @classmethod
    def indexToPos(self, index, min_pos, resolution):
        return float(index * resolution + min_pos)

class DijkstraPlanner:
    def __init__(self, start, goal, obstacles, config):
        self.start_ = start  # 起点
        self.goal_ = goal  # 终点
        self.obstacles_ = obstacles  # 障碍物信息
        self.config_ = config  # 设置
        self.__buildGridMap()  # 构造栅格地图
    
    # 构造栅格地图
    def __buildGridMap(self):
        # 栅格地图边界
        ox, oy = Tools.departPoints(self.obstacles_)
        self.gridmap_minx_ = min(ox)
        self.gridmap_maxx_ = max(ox)
        self.gridmap_miny_ = min(oy)
        self.gridmap_maxy_ = max(oy)
        print("gridmap min pos_x: ", self.gridmap_minx_)
        print("gridmap max pos_x: ", self.gridmap_maxx_)
        print("gridmap min pos_y: ", self.gridmap_miny_)
        print("gridmap max pos_y: ", self.gridmap_maxy_)
        self.gridmap_nxwidth_ = Tools.posToIndex(self.gridmap_maxx_, self.gridmap_minx_, self.config_.resolution_)
        self.gridmap_nywidth_ = Tools.posToIndex(self.gridmap_maxy_, self.gridmap_miny_, self.config_.resolution_)
        print("gridmap xwidth index: ", self.gridmap_nxwidth_)
        print("gridmap ywidth index: ", self.gridmap_nywidth_)
        # 初始化栅格地图
        gridmap = [[True for j in range(0, self.gridmap_nywidth_)] for i in range(0, self.gridmap_nxwidth_)]
        # 判断栅格地图中每一个栅格是否为障碍点
        for i in range(0, self.gridmap_nxwidth_):
            for j in range(0, self.gridmap_nywidth_):
                gridmap_point = Point(Tools.indexToPos(i, self.gridmap_minx_, self.config_.resolution_), Tools.indexToPos(j, self.gridmap_miny_, self.config_.resolution_))
                for obstacle in self.obstacles_:
                    if gridmap_point - obstacle <= self.config_.robot_size_:
                        gridmap[i][j] = False
                        break;
        self.gridmap_ = gridmap

    # 进行规划
    def planning(self):
        # 创建起始节点和目标节点
        nstart = Node(Tools.posToIndex(self.start_.x_, self.gridmap_minx_, self.config_.resolution_), Tools.posToIndex(self.start_.y_, self.gridmap_miny_, self.config_.resolution_), 0, -1)
        ngoal = Node(Tools.posToIndex(self.goal_.x_, self.gridmap_minx_, self.config_.resolution_), Tools.posToIndex(self.goal_.y_, self.gridmap_miny_, self.config_.resolution_), 0, -1)
        # 初始化搜索节点集合和结束搜索节点集合
        openset = dict()
        closeset = dict()
        openset[nstart.ny_ * self.gridmap_nxwidth_ + nstart.nx_] = nstart
        # 开始搜索
        while True:
            # 得到当前搜索节点
            current_node_id = min(openset, key=lambda o: openset[o].cost_)
            current_node = openset[current_node_id]
            # 可视化当前搜索点
            plt.plot(Tools.indexToPos(current_node.nx_, self.gridmap_minx_, self.config_.resolution_), Tools.indexToPos(current_node.ny_, self.gridmap_miny_, self.config_.resolution_), "xc")
            if len(closeset.keys()) % 10 == 0:
                plt.pause(0.001)
            # 判断当前搜索节点是否为目标点
            if current_node == ngoal:
                # 找到目标点
                print("Goal Finded")
                ngoal.prev_id_ = current_node.prev_id_
                ngoal.cost_ = current_node.cost_
                break
            else:
                # 未能找到目标点
                # 从搜索集合中删除节点并加入搜索完成集合
                del openset[current_node_id]
                closeset[current_node_id] = current_node
                # 遍历当前搜索点采取特定行为后的节点是否可以加入搜索集合
                for i in range(0, len(self.config_.motion_)):
                    motion = self.config_.motion_[i]
                    next_node = Node(current_node.nx_ + motion.x_mv_, current_node.ny_ + motion.y_mv_, current_node.cost_ + motion.cost_, current_node_id)
                    next_node_id = next_node.nx_ + next_node.ny_ * self.gridmap_nxwidth_
                    # 判断下一个点是否可以为搜索点
                    # 条件一不是搜索完成的点
                    if next_node_id in closeset:
                        continue
                    # 条件二不是无效点
                    if not self.verifyNode(next_node):
                        continue
                    # 条件三不在openset中或在openset中的代价更大
                    if not next_node_id in openset:
                        openset[next_node_id] = next_node
                    else:
                        if openset[next_node_id].cost_ >= next_node.cost_:
                            openset[next_node_id] = next_node
        # 由目标点得到最终路径
        pgoal = Point(Tools.indexToPos(ngoal.nx_, self.gridmap_minx_, self.config_.resolution_), Tools.indexToPos(ngoal.ny_, self.gridmap_miny_, self.config_.resolution_))
        final_path = [pgoal]
        prev_node_id = ngoal.prev_id_
        while prev_node_id != -1:
            prev_node = closeset[prev_node_id]
            pprev_node = Point(Tools.indexToPos(prev_node.nx_, self.gridmap_minx_, self.config_.resolution_), Tools.indexToPos(prev_node.ny_, self.gridmap_miny_, self.config_.resolution_))
            final_path.append(pprev_node)
            prev_node_id = prev_node.prev_id_
        final_path.reverse()
        return final_path

    # 有效性检测
    def verifyNode(self, node):
        px = Tools.indexToPos(node.nx_, self.gridmap_minx_, self.config_.resolution_)
        py = Tools.indexToPos(node.ny_, self.gridmap_miny_, self.config_.resolution_)
        # 判断是否在范围内
        if px >= self.gridmap_minx_ and px < self.gridmap_maxx_ and py >= self.gridmap_miny_ and py < self.gridmap_maxy_:
            # 判断是否存在障碍物
            if self.gridmap_[node.nx_][node.ny_]:
                return True
            else:
                return False
        else:
            return False
    
# 主函数
def main():
    # 设定起点和终点
    start = Point(-5.0, -5.0)
    goal = Point(50.0, 50.0)
    # 得到初始设定
    config = Config()
    # 构建障碍物
    obstacles = []
    # 边界障碍物
    for ox in range(-10, 61):
        for oy in range(-10, 61):
            if ox == -10:
                obstacles.append(Point(ox, oy))
            elif ox == 60:
                obstacles.append(Point(ox, oy))
            elif oy == -10:
                obstacles.append(Point(ox, oy))
            elif oy == 60:
                obstacles.append(Point(ox, oy))
            elif ox == 20 and oy < 40:
                obstacles.append(Point(ox, oy))
            elif ox == 40 and oy > 20:
                obstacles.append(Point(ox, oy))
    # 对当前信息进行可视化
    obstalces_x, obstacles_y = Tools.departPoints(obstacles)
    plt.plot(obstalces_x, obstacles_y, ".k")
    plt.plot(start.x_, start.y_, "og")
    plt.plot(goal.x_, goal.y_, "xb")
    plt.grid(True)
    plt.axis("equal")
    # 构建Dijkstra规划器
    dijkstra_planner = DijkstraPlanner(start, goal, obstacles, config)
    planned_path = dijkstra_planner.planning()
    # 可视化最终路径
    path_x, path_y = Tools.departPoints(planned_path)
    plt.plot(path_x, path_y, "-r")
    plt.show()

if __name__ == "__main__":
    main()