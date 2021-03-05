#! /usr/bin/python3
#! -*- coding:utf-8 -*-

"""

Hybrid Astar
author: flztiii

"""

import numpy as np
import scipy.linalg
import scipy.spatial
import matplotlib.pyplot as plt

# 全局变量
XY_GRID_RESOLUTION = 2.0  # 栅格地图x,y分辨率[m]
YAW_GRID_RESOLUTION = np.deg2rad(15.0)  # 栅格地图角度分辨率[rad]
MOTION_RESOLUTION = 0.1  # 路径采样点间隔[m]
N_STEER = 20.0  # 转向角采样数量
VR = 1.0  # 机器人半径

SB_COST = 100.0  # 从前进切换到倒车的损失
BACK_COST = 5.0  # 倒车行驶的损失
STEER_CHANGE_COST = 5.0  # 朝向变化损失
STEER_COST = 1.0  # 朝向损失
H_COST = 1.0  # 启发函数损失

# 节点
class Node:
    def __init__(self, xind, yind, yawind, direction, xlist, ylist, yawlist, directions, steer, cost, pind):
        self.xind_ = xind  # 栅格地图x坐标
        self.yind_ = yind  # 栅格地图y坐标
        self.yawind_ = yawind  # 栅格地图yaw坐标
        self.direction_ = direction  # 当前点朝向
        self.xlist_ = xlist  # 从上一个点到当前点的路径x坐标
        self.ylist_ = ylist  # 从上一个点到当前点的路径y坐标
        self.yawlist_ = yawlist  # 从上一个点到当前点的路径yaw坐标
        self.directions_ = directions  # 从上一个点到当前点的路径朝向
        self.steer_ = steer  # 转角变化率
        self.cost_ = cost  # 当前点损失
        self.pind_ = pind  # 上一个点对应下标


# 栅格地图信息
class GridMapInfo:
    def __init__(self, ox, oy, reso_xy, reso_yaw):
        self.min_x_ = round(min(ox) / reso_xy)  # 栅格地图最小x下标
        self.max_x_ = round(max(ox) / reso_xy)  # 栅格地图最大x下标
        self.min_y_ = round(min(oy) / reso_xy)  # 栅格地图最小y下标
        self.max_y_ = round(max(oy) / reso_xy)  # 栅格地图最大y下标
        self.wx_ = self.max_x_ - self.min_x_  # 栅格地图长度
        self.wy_ = self.max_y_ - self.max_y_  # 栅格地图宽度
        self.min_yaw_ = round(- np.pi / reso_yaw) - 1  # 栅格地图最小yaw下标
        self.max_yaw_ = round(np.pi / reso_yaw)  # 栅格地图最大yaw下标
        self.wyaw_ = self.max_yaw_ - self.min_yaw_  # 栅格地图高度
    
# 障碍物信息
class ObsKDTree:
    def __init__(self, ox, oy):
        obstacles = np.vstack((ox, oy)).T
        self.kdtree = scipy.spatial.cKDTree(obstacles)
    
    # 寻找给定坐标最近邻k个障碍物下标
    def findKNeighbor(self, inp, k):
        return self.kdtree.query(inp, k)
    
    # 寻找给定坐标一定范围内的障碍物下标
    def findRangeNeighbor(self, inp, range):
        return self.kdtree.query_ball_point(inp, range)
        

# 混合a星函数
def hybridAstarPlanning(start, goal, ox, oy, reso_xy, reso_yaw):
    # 第一步,构建栅格地图相关信息,障碍物相关信息
    grid_map_info = GridMapInfo(ox, oy, reso_xy, reso_yaw)
    obs_kdtree = ObsKDTree(ox, oy)

    # 第二步,构造起点和终点节点
    start_node = Node(round(start[0] / reso_xy), round(start[1] / reso_xy), round(start[2] / reso_yaw), True, [start[0]], [start[1]], [start[2]], [True], 0.0, 0.0, None)

    goal_node = Node(round(goal[0] / reso_xy), round(goal[1] / reso_xy), round(goal[2] / reso_yaw), True, [goal[0]], [goal[1]], [goal[2]], [True], 0.0, None, None)

    # 第三步,构建搜索的开集合和闭集合,开集合为等待搜索节点集合,闭集合为搜索完成节点集合
    open_set, close_set = dict(), dict()
    # 将起始节点加入开集合
    open_set[Tools.calc_index(start_node, grid_map_info)] = start_node

    # 第四步,开始搜索
    while True:
        # 判断开集合是否为空
        if not open_set:
            # 没能找到终点
            print('Cannot reach goal')
            exit(0)
        # 从开集合中找到损失加上启发最小的
        min_cost_index = min(open_set, key = lambda o: open_set[o].cost_ + Tools.calc_heuristic(open_set[o], goal_node, H_COST))
        current_node = open_set[min_cost_index]
        # 将当前节点从开集合中删除,并加入到闭集合中
        del open_set[min_cost_index]
        close_set[min_cost_index] = current_node
        # 进行可视化
        if len(close_set.keys()) % 1 == 0:
            plt.plot(current_node.xlist_, current_node.ylist_, "grey")
            plt.pause(0.001)
        # 判断当前节点是否存在到达终点的路径
        path_x, path_y, path_yaw = Tools.exitPathToGoal(current_node, goal_node, obs_kdtree, ox, oy)
        if path_x is not None:
            # 找到了终点的路径
            # 重新构造终点节点
            goal_node.xlist_ = path_x[1:]
            goal_node.ylist_ = path_y[1:]
            goal_node.yawlist_ = path_yaw[1:]
            goal_node.pind_ = min_cost_index
            break
        # 如果当前节点没有能够直接到达终点的路径，则对当前节点的邻居进行搜索
        for neighbor_node in Tools.getNeighbors(current_node, grid_map_info, reso_xy, reso_yaw):
            # 判断邻居节点是否无碰撞
            if Tools.pathCollision(neighbor_node.xlist_, neighbor_node.ylist_, neighbor_node.yawlist_, obs_kdtree, ox, oy):
                # 发生碰撞直接无视
                continue
            # 如果没有发生碰撞,判断节点是否在闭集合内
            neighbor_index = Tools.calc_index(neighbor_node, grid_map_info)
            if neighbor_index in close_set:
                # 在闭集合内,说明该点已经被搜索过了,不需要继续进行搜索
                continue
            # 如果没有在闭集合内,判断是否在开集合内
            if neighbor_index in open_set:
                # 在开集合内
                # 判断开集合内对应节点的代价
                if neighbor_node.cost_ < open_set[neighbor_index].cost_:
                    # 如果当前邻居代价更低,修改为当前邻居节点
                    open_set[neighbor_index] = neighbor_node
            else:
                # 不在开集合内
                # 将邻居节点加入开集合,等待进行搜索
                open_set[neighbor_index] = neighbor_node
    # 根据闭集合,找到最终路径
    path_x, path_y, path_yaw = Tools.find_final_path(goal_node, close_set)
    return path_x, path_y, path_yaw

# 工具类
class Tools:
    # 常量
    WB = 3.  # rear to front wheel
    W = 2.  # width of car
    LF = 3.3  # distance from rear to vehicle front end
    LB = 1.0  # distance from rear to vehicle back end
    MAX_STEER = 0.6  # [rad] maximum steering angle

    WBUBBLE_DIST = (LF - LB) / 2.0
    WBUBBLE_R = np.sqrt(((LF + LB) / 2.0)**2 + 1)

    # vehicle rectangle verticles
    VRX = [LF, LF, -LB, -LB, LF]
    VRY = [W / 2, -W / 2, -W / 2, W / 2, W / 2]

    def __init__(self):
        pass
    
    # 计算栅格节点启发函数
    @classmethod
    def calc_heuristic(self, current_node, goal_node, weight):
        return np.sqrt((current_node.xlist_[-1] - goal_node.xlist_[-1]) ** 2 + (current_node.ylist_[-1] - goal_node.ylist_[-1]) ** 2) * weight
    
    # 计算栅格节点对应下标
    @classmethod
    def calc_index(self, node, grid_map_info):
        return node.xind_ + node.yind_ * grid_map_info.wx_ + node.yawind_ * grid_map_info.wx_ * grid_map_info.wy_;
    
    # 判断是否存在到终点的曲线
    @classmethod
    def exitPathToGoal(self, current_node, goal_node, obs_kdtree, ox, oy):
        # 规划出从当前点到终点的三次样条曲线
        cublic_spline = CubicSpline(current_node.xlist_[-1], current_node.ylist_[-1], current_node.yawlist_[-1], goal_node.xlist_[-1], goal_node.ylist_[-1], goal_node.yawlist_[-1])
        path_x, path_y, path_yaw = cublic_spline.getPath()
        # 判断路径是否与障碍物发生碰撞
        if self.pathCollision(path_x, path_y, path_yaw, obs_kdtree, ox, oy):
            # 发生碰撞
            return None, None, None
        else:
            # 没有发生碰撞
            return path_x, path_y, path_yaw
    
    # 判断路径是否与障碍物发生碰撞
    @classmethod
    def pathCollision(self, path_x, path_y, path_yaw, obs_kdtree, ox, oy):
        for x, y, yaw in zip(path_x, path_y, path_yaw):
            # 计算后轴中心在路径上是几何中心坐标
            cx = x + self.WBUBBLE_DIST * np.cos(yaw)
            cy = y + self.WBUBBLE_DIST * np.sin(yaw)
            # 判断障碍物离车辆距离是否在半车长以内的
            ids = obs_kdtree.findRangeNeighbor([cx, cy], self.WBUBBLE_R)
            if not ids:
                continue
            # 判断车辆是否与障碍物发生碰撞
            if not self.rectangleCheck(x, y, yaw, [ox[i] for i in ids], [oy[i] for i in ids]):
                return True
        return False
    
    # 判断矩形和障碍物是否发生碰撞
    @classmethod
    def rectangleCheck(self, x, y, yaw, ox, oy):
        # transform obstacles to base link frame
        c, s = np.cos(-yaw), np.sin(-yaw)
        for iox, ioy in zip(ox, oy):
            tx = iox - x
            ty = ioy - y
            rx = c * tx - s * ty
            ry = s * tx + c * ty

            if not (rx > self.LF or rx < -self.LB or ry > self.W / 2.0 or ry < -self.W / 2.0):
                return False  # no collision

        return True  # collision
    
    # 设置车辆可以采取的行为
    @classmethod
    def motions(self):
        for steer in np.concatenate((np.linspace(-self.MAX_STEER, self.MAX_STEER, N_STEER),[0.0])):
            for d in [1.0, -1.0]:
                yield [d, steer]

    # 计算当前节点的邻居节点
    @classmethod
    def getNeighbors(self, current_node, grid_map_info, reso_xy, reso_yaw):
        # 运动即将走的路程
        arc_l = 1.5 * XY_GRID_RESOLUTION
        for motion in self.motions():
            # 计算根据该行为得到的路径
            path_x, path_y, path_yaw = [], [], []
            x, y, yaw = current_node.xlist_[-1], current_node.ylist_[-1], current_node.yawlist_[-1]
            for distance in np.arange(0.0, arc_l + MOTION_RESOLUTION, MOTION_RESOLUTION):
                x, y, yaw = self.move(x, y, yaw, MOTION_RESOLUTION * motion[0], motion[1])
                path_x.append(x)
                path_y.append(y)
                path_yaw.append(yaw)
            # 计算路径的损失
            cost = arc_l + STEER_COST * abs(motion[1]) + STEER_CHANGE_COST * abs(current_node.steer_ - motion[1]) + current_node.cost_;
            # 构建新的节点
            neighbor_node = Node(round(path_x[-1] / reso_xy), round(path_y[-1] / reso_xy), round(path_yaw[-1] / reso_yaw), True, path_x, path_y, path_yaw, [True], motion[1], cost, self.calc_index(current_node, grid_map_info))
            # 判断节点是否有效
            if self.verifyNode(neighbor_node, grid_map_info):
                yield neighbor_node
    
    # 计算移动后下一个点的位置
    @classmethod
    def move(self, x, y, yaw, distance, steer):
        x = x + distance * np.cos(yaw)
        y = y + distance * np.sin(yaw)
        yaw = yaw + self.pi_2_pi(distance * np.tan(steer) / self.WB)
        return x, y, yaw

    # 角度转化到[-pi,pi]
    @classmethod
    def pi_2_pi(self, theta):
        return (theta + np.pi) % (2 * np.pi) - np.pi
    
    # 判断节点是否有效
    @classmethod
    def verifyNode(self, node, grid_map_info):
        if node.xind_ >= grid_map_info.min_x_ and node.xind_ <= grid_map_info.max_x_:
            if node.yind_ >= grid_map_info.min_y_ and node.yind_ <= grid_map_info.max_y_:
                return True
        return False
    
    # 找到最终路径
    @classmethod
    def find_final_path(self, goal_node, close_set):
        path_x, path_y, path_yaw = [], [], []
        current_node = goal_node
        while current_node.pind_ is not None:
            path_x = current_node.xlist_ + path_x
            path_y = current_node.ylist_ + path_y
            path_yaw = current_node.yawlist_ + path_yaw
            current_node = close_set[current_node.pind_]
        return path_x, path_y, path_yaw

    # 绘制车辆
    @classmethod
    def plot_car(self, x, y, yaw):
        car_color = '-k'
        c, s = np.cos(yaw), np.sin(yaw)

        car_outline_x, car_outline_y = [], []
        for rx, ry in zip(self.VRX, self.VRY):
            tx = c * rx - s * ry + x
            ty = s * rx + c * ry + y
            car_outline_x.append(tx)
            car_outline_y.append(ty)

        arrow_x, arrow_y, arrow_yaw = c * 1.5 + x, s * 1.5 + y, yaw
        plot_arrow(arrow_x, arrow_y, arrow_yaw)

        plt.plot(car_outline_x, car_outline_y, car_color)

# 三次曲线插值
class CubicSpline:
    def __init__(self, sx, sy, syaw, gx, gy, gyaw):
        self.arc_l_ = np.sqrt((sx - gx) ** 2 + (sy - gy) ** 2)
        # 求解x参数
        self.x_param_ = self.__calcParam(sx, np.cos(syaw), gx, np.cos(gyaw), self.arc_l_)
        self.y_param_ = self.__calcParam(sy, np.sin(syaw), gy, np.sin(gyaw), self.arc_l_)

    # 计算参量
    def __calcParam(self, start, start_derivation, goal, goal_derivation, arc_l):
        a0 = start
        a1 = start_derivation
        A = np.array([[arc_l ** 2, arc_l ** 3], [2. * arc_l, 3. * arc_l ** 2]])
        B = np.array([goal - a0 - a1 * arc_l, goal_derivation - a1])
        result = scipy.linalg.solve(A, B)
        return np.array([a0, a1, result[0], result[1]])
    
    # 获得曲线
    def getPath(self):
        samples = np.arange(0.0, self.arc_l_ + MOTION_RESOLUTION, MOTION_RESOLUTION)
        path_x, path_y, path_yaw = [], [], []
        for sample in samples:
            x = np.dot(self.x_param_, np.array([1.0, sample, sample ** 2, sample ** 3]))
            y = np.dot(self.y_param_, np.array([1.0, sample, sample ** 2, sample ** 3]))
            yaw = np.arctan2(np.dot(self.y_param_, np.array([0.0, 1.0, 2. * sample, 3. * sample ** 2])), np.dot(self.x_param_, np.array([0.0, 1.0, 2. * sample, 3. * sample ** 2])))
            path_x.append(x)
            path_y.append(y)
            path_yaw.append(yaw)
        return path_x, path_y, path_yaw
    

def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    """Plot arrow."""
    if not isinstance(x, float):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * np.cos(yaw), length * np.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width, alpha=0.4)

# 主函数
def main():
    print("Start Hybrid A* planning")
    
    # 初始化障碍物信息
    ox, oy = [], []

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
    
    # 可视化障碍物
    plt.plot(ox, oy, ".k")

    # 初始化起点信息
    start = [10.0, 10.0, np.deg2rad(90.0)]
    # 初始化终点信息
    goal = [50.0, 50.0, np.deg2rad(90.0)]

    # 可视化起点和终点
    plot_arrow(start[0], start[1], start[2])
    plot_arrow(goal[0], goal[1], goal[2])

    # 开始进行规划
    path_x, path_y, path_yaw = hybridAstarPlanning(start, goal, ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)

    # 可视化路径
    plt.plot(path_x, path_y, 'g')
    # 可视化车辆运动
    for x, y, yaw in zip(path_x, path_y, path_yaw):
        plt.cla()
        plt.plot(ox, oy, ".k")
        plt.plot(path_x, path_y, "-r", label="Hybrid A* path")
        plt.grid(True)
        plt.axis("equal")
        Tools.plot_car(x, y, yaw)
        plt.pause(0.0001)

    plt.show()

# 测试函数
def test():
    cubic_s = CubicSpline(0.0, 0.0, 0.0, 10.0, 10.0, 0.0)
    path_x, path_y, path_yaw = cubic_s.getPath()
    plt.figure()
    plt.plot(path_x, path_y)
    indexes = range(0, len(path_yaw))
    plt.figure()
    plt.plot(indexes, path_yaw)
    plt.show()

if __name__ == "__main__":
    main()