#! /usr/bin/python3
#! -*- coding: utf-8 -*-

"""

grid astar with smoothing

author: flztiii

"""

import matplotlib.pyplot as plt
import numpy as np
import math
import copy

VISUALIZE = True  # 可视化参数

# 点
class Point:
    def __init__(self, x, y):
        self.x_ = x  # x轴坐标
        self.y_ = y  # y轴坐标
    
    # 重载减法操作
    def __sub__(self, point):
        return math.sqrt((self.x_ - point.x_) ** 2 + (self.y_ - point.y_) ** 2)

    # 重载等于
    def __eq__(self, point):
        if self.x_ == point.x_ and self.y_ == point.y_:
            return True
        else:
            return False

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
            Motion(1, 0), 
            Motion(0, 1),
            Motion(-1, 0),
            Motion(0, -1),
            Motion(-1, -1),
            Motion(-1, 1), 
            Motion(1, -1), 
            Motion(1, 1)
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

    # 计算启发
    @classmethod
    def calc_heuristic(self, ngoal, node, weight = 1.0):
        return weight * math.sqrt((ngoal.nx_ - node.nx_) ** 2 + (ngoal.ny_ - node.ny_) ** 2)
    
    # 点转向量
    @classmethod
    def pointToArray(self, point):
        return np.array([point.x_, point.y_])
    
    # 路径转向量组
    @classmethod
    def pathToArraySet(self, path):
        result = []
        for point in path:
            result.append([point.x_, point.y_])
        return np.array(result)
    
    # 向量组转路径
    @classmethod
    def arraySetToPath(self, array_set):
        path = []
        for array in array_set:
            point = Point(array[0], array[1])
            path.append(point)
        return path

    # 向量之间求垂直
    @classmethod
    def orz(self, vector1, vector2):
        return vector1 - np.dot(vector1.T, vector2) * vector2 / (np.linalg.norm(vector2) ** 2)

    # 判断路径中是否存在该点
    @classmethod
    def isInPointSet(self, point, point_set):
        for p in point_set:
            if p == point:
                return True
        return False

    # 高斯函数
    @classmethod
    def gaussian(self, x, mu, sigma):
        return (1.0 / np.sqrt(2.0 * np.pi * sigma ** 2)) * np.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))

class AstarPlanner:
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
            current_node_id = min(openset, key=lambda o: openset[o].cost_ + Tools.calc_heuristic(ngoal, openset[o]))
            current_node = openset[current_node_id]
            # 可视化当前搜索点
            if VISUALIZE:
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
                        if openset[next_node_id].cost_ > next_node.cost_:
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

class Smoother:
    def __init__(self):
        self.obs_weight_ = 1.0  # 障碍物损失权重
        self.cur_weight_ = 0.0001  # 曲率损失权重
        self.smoo_weight_ = 0.1  # 平滑损失权重
        self.obs_max_distance_ = 5.0  # 离障碍物最大距离
        self.max_curvature_ = 0.1  # 最大曲率
        self.visualiation_ = True  # 是否可视化

    # 路径平滑
    def smoothing(self, path, obstacles):
        # 路径平滑考虑到三个因素,离障碍物的距离,曲率大小和采样点间隔变化
        smoothed_path = copy.deepcopy(path)
        # 采用的方法是梯度下降法
        # 首先确定最大迭代次数
        max_iter = 50
        # 当前迭代次数
        iter_num = 0
        # 损失记录器
        obs_cost_recorder, curvature_cost_recorder, smooth_cost_recorder = [], [], []
        # 路径上点的数量
        path_length = len(path)
        # 开始迭代
        while iter_num < max_iter:
            # 路径计算损失
            # 更新梯度
            gradient = np.zeros((path_length, 2))
            # 首先计算障碍物带来的梯度
            obstacle_gradient = np.zeros((path_length, 2))
            for i in range(1, path_length - 1):
                # 首先从路径中取出1个点
                point = smoothed_path[i]
                # 计算障碍物梯度
                obstacle_gradient_item = self.__calcObstacleItem(point, obstacles)
                # 加入总梯度
                obstacle_gradient[i] += obstacle_gradient_item
            # 计算平滑带来的梯度
            smoo_gradient = np.zeros((path_length, 2))
            for i in range(1, path_length - 1):
                # 首先从路径中取出3个点
                point_m1 = Tools.pointToArray(smoothed_path[i - 1])
                point = Tools.pointToArray(smoothed_path[i])
                point_p1 = Tools.pointToArray(smoothed_path[i + 1])
                # 计算平滑度梯度
                smoo_gradient_item_m1, smoo_gradient_item, smoo_gradient_item_p1 = self.__calcSmoothItem(point_m1, point, point_p1)
                # 加入总梯度
                smoo_gradient[i - 1] += smoo_gradient_item_m1
                smoo_gradient[i] += smoo_gradient_item
                smoo_gradient[i + 1] += smoo_gradient_item_p1
            # 计算曲率带来的梯度
            curvature_gradient = np.zeros((path_length, 2))
            for i in range(1, path_length - 1):
                point_m1 = Tools.pointToArray(smoothed_path[i - 1])
                point = Tools.pointToArray(smoothed_path[i])
                point_p1 = Tools.pointToArray(smoothed_path[i + 1])
                # 计算曲率梯度
                cur_gradient_item_m1, cur_gradient_item, cur_gradient_item_p1 = self.__calcCurvatureItem(point_m1, point, point_p1)
                # 加入总梯度
                curvature_gradient[i - 1] += cur_gradient_item_m1
                curvature_gradient[i] += cur_gradient_item
                curvature_gradient[i + 1] += cur_gradient_item_p1

            # 计算学习率
            lrs = [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
            lr_obs, lr_cur, lr_smo = 0.0, 0.0, 0.0
            # 计算障碍物学习率
            values = []
            for lr in lrs:
                value = self.__calcObstacleCost(Tools.arraySetToPath(Tools.pathToArraySet(smoothed_path) - lr * obstacle_gradient), obstacles)
                values.append(value)
            lr_obs = lrs[values.index(min(values))]
            # 计算曲率学习率
            values = []
            for lr in lrs:
                value = self.__calcCurvatureCost(Tools.arraySetToPath(Tools.pathToArraySet(smoothed_path) - lr * curvature_gradient))
                values.append(value)
            lr_cur = lrs[values.index(min(values))]
            # 计算平滑度的学习率
            values = []
            for lr in lrs:
                value = self.__calcSmoothCost(Tools.arraySetToPath(Tools.pathToArraySet(smoothed_path) - lr * smoo_gradient))
                values.append(value)
            lr_smo = lrs[values.index(min(values))]
            # 计算总曲率
            gradient = self.obs_weight_ * lr_obs * obstacle_gradient + self.cur_weight_ * lr_cur * curvature_gradient + self.smoo_weight_ * lr_smo * smoo_gradient

            # # 计算曲率对比梯度
            # compare_gradient = self.__calcCurvatureItemCompare(smoothed_path)
            # gradient = 3.0 * compare_gradient
            # for i in range(1, path_length -1):
            #     print('raw:', gradient[i], 'compare:', compare_gradient[i])

            # 更新路径(起点和终点不变)
            for i in range(1, path_length - 1):
                smoothed_path[i].x_ += - gradient[i][0]
                smoothed_path[i].y_ += - gradient[i][1]
            
            # 计算当前迭代平滑路径的损失
            # 曲率损失
            curvature_cost_recorder.append(self.__calcCurvatureCost(smoothed_path))
            # 障碍物损失
            obs_cost_recorder.append(self.__calcObstacleCost(smoothed_path, obstacles))
            # 平滑损失
            smooth_cost_recorder.append(self.__calcSmoothCost(smoothed_path))

            # 更新计数器
            iter_num += 1

            # 可视化平滑路径
            if iter_num % 1 == 0:
                plt.cla()
                # 可视化路径和环境
                path_x, path_y = Tools.departPoints(path)
                plt.plot(path_x, path_y, "-r")
                obstalces_x, obstacles_y = Tools.departPoints(obstacles)
                plt.plot(obstalces_x, obstacles_y, ".k")
                plt.grid(b=True,which='major',axis='both',alpha= 0.5,color='skyblue',linestyle='--',linewidth=2)
                plt.axis("equal")
                spath_x, spath_y = Tools.departPoints(smoothed_path)
                plt.plot(spath_x, spath_y, "-g")
                # 可视化损失变化
                # fig = plt.figure()
                # plt.plot(np.linspace(0, len(curvature_cost_recorder), len(curvature_cost_recorder)), curvature_cost_recorder)
                plt.pause(0.1)
                # plt.close('all')

            # # 判断是否结束
            # if np.linalg.norm(gradient) < 0.001:
            #     print('break')
            #     break
        print('optimization finished')
        return smoothed_path

    # 角度滤波(双边滤波器)
    def yawBlur(self, path, sigma_s, sigma_yaw):
        # 最终结果
        blurred_yaw = np.zeros(len(path))
        # 首先计算路径每个点的arc length
        arc_lengths = [0.0]
        for i in range(1, len(path)):
            distance = path[i] - path[i - 1]
            arc_lengths.append(arc_lengths[-1] + distance)
        assert len(arc_lengths) == len(path)
        # 计算每个点的朝向
        yaws = []
        for i in range(0, len(path) - 1):
            yaw = np.arctan2(path[i + 1].y_ - path[i].y_, path[i + 1].x_ - path[i].x_)
            yaws.append(yaw)
        final_yaw = yaws[-1] + (yaws[-1] - yaws[-2]) / (arc_lengths[-2] - arc_lengths[-3]) * (arc_lengths[-1] - arc_lengths[-2])
        yaws.append(final_yaw)
        assert len(yaws) == len(path)
        # 确定双边滤波器的窗长度
        window_size = sigma_s * 5.0
        # 开始进行滤波
        # 遍历每一个点
        for i, point in enumerate(path):
            # 求以当前点为终点,在窗内的点
            neighbor_index = [i]
            p_enable, m_enable = True, True
            for j in range(1, len(path)):
                # 判断邻居是否有效
                if i + j in range(0, len(path)) and p_enable:
                    #邻居有效,计算当前点到邻居的距离是否小于窗长
                    if path[i + j] - path[i] < window_size:
                        neighbor_index.append(i + j)
                    else:
                        p_enable = False
                else:
                    p_enable = False
                # 判断邻居是否有效
                if i - j in range(0, len(path)) and m_enable:
                    #邻居有效,计算当前点到邻居的距离是否小于窗长
                    if path[i - j] - path[i] < window_size:
                        neighbor_index.append(i - j)
                    else:
                        m_enable = False
                else:
                    m_enable = False
                # 判断是否结束循环
                if (not p_enable) and (not m_enable):
                    break
            neighbor_index = sorted(neighbor_index)
            assert len(neighbor_index) > 0
            # 找到邻居后,进行双边滤波
            I, W =0.0, 0.0
            # 计算权重
            for index in neighbor_index:
                gs = Tools.gaussian(arc_lengths[index], arc_lengths[i], sigma_s)
                gy = Tools.gaussian(yaws[index], yaws[i], sigma_yaw)
                weight = gs * gy
                I += yaws[index] * weight
                W += weight
            blurred_yaw[i] = I / W
        return blurred_yaw


    # 计算障碍物梯度
    def __calcObstacleItem(self, point, obstacles):
        # 遍历计算全部障碍物离给出点的距离
        min_distance = 1000000.0
        index = 0
        for i, obstacle in enumerate(obstacles):
            distance = point - obstacle
            if distance < min_distance:
                min_distance = distance
                index = i
        gradient = -1.0 / (min_distance ** 3) * np.array([point.x_ - obstacles[index].x_, point.y_ - obstacles[index].y_])
        return gradient
        # if min_distance < self.obs_max_distance_:
        #     # 如果小于离障碍物最大距离
        #     gradient = 2.0 * (min_distance - self.obs_max_distance_) / min_distance * np.array([point.x_ - obstacles[index].x_, point.y_ - obstacles[index].y_])
        #     return gradient
        # else:
        #     # 如果大于离障碍物最大距离,梯度为0
        #     gradient = np.array([0.0, 0.0])
        #     return gradient

    # 计算平滑梯度
    def __calcSmoothItem(self, point_m1, point, point_p1):
        gradient_m1 = 2.0 * (point_m1 + point_p1 - 2.0 * point)
        gradient_p1 = 2.0 * (point_m1 + point_p1 - 2.0 * point)
        gradient = -4.0 * (point_m1 + point_p1 - 2.0 * point)
        return gradient_m1, gradient, gradient_p1
    
    # 计算曲率梯度
    def __calcCurvatureItem(self, point_m1, point, point_p1):
        # 计算第前一个点到中间点的向量
        delta_x = point - point_m1
        abs_delta_x = np.linalg.norm(delta_x)
        # 计算中间点到后一个点的向量
        delta_px = point_p1 - point
        abs_delta_px = np.linalg.norm(delta_px)
        # 计算角度变化量
        delta_phi = np.arccos(np.dot(delta_x.T, delta_px) / (abs_delta_x * abs_delta_px))
        # 计算曲率
        curvature = delta_phi / abs_delta_x
        # 判断曲率是否大于最大曲率
        if curvature > self.max_curvature_:
            # 大于最大曲率,梯度不为0
            # 计算梯度
            u = 1. / abs_delta_x * -1. / (1 - np.cos(delta_phi) ** 2) ** 0.5
            p1 = Tools.orz(delta_x, -delta_px) / (abs_delta_x * abs_delta_px)
            p2 = Tools.orz(-delta_px, delta_x) / (abs_delta_x * abs_delta_px)
            gradient = 2.0 * (curvature - self.max_curvature_) * (u * (-p1 - p2) - delta_phi / (abs_delta_x ** 3) * delta_x)
            gradient_p1 = 2.0 * (curvature - self.max_curvature_) * (u * p2 + delta_phi / (abs_delta_x ** 3) * delta_x)
            gradient_m1 = 2.0 * (curvature - self.max_curvature_) * (u * p1)
            return gradient_m1, gradient, gradient_p1
        else:
            # 小于最大曲率,梯度为0
            gradient = np.array([0.0, 0.0])
            gradient_m1 = np.array([0.0, 0.0])
            gradient_p1 = np.array([0.0, 0.0])
            return gradient_m1, gradient, gradient_p1

    # 计算障碍物损失函数
    def __calcObstacleCost(self, path, obstacles):
        cost = 0.0
        for i in range(1, len(path) - 1):
            point = path[i]
            min_distance = 1000000.0
            for obstacle in obstacles:
                distance = point - obstacle
                if distance < min_distance:
                    min_distance = distance
            # if min_distance < self.obs_max_distance_:
            cost += 1.0 / (min_distance ** 2)
        return cost

    # 计算曲率损失函数
    def __calcCurvatureCost(self, path):
        cost = 0.0
        for i in range(1, len(path) - 1):
            point_m1 = Tools.pointToArray(path[i - 1])
            point = Tools.pointToArray(path[i])
            point_p1 = Tools.pointToArray(path[i + 1])
            delta_x = point - point_m1
            abs_delta_x = np.linalg.norm(delta_x)
            delta_px = point_p1 - point
            abs_delta_px = np.linalg.norm(delta_px)
            delta_phi = np.arccos(np.dot(delta_x.T, delta_px) / (abs_delta_x * abs_delta_px))
            # 计算曲率
            curvature = delta_phi / abs_delta_x
            if curvature > self.max_curvature_:
                cost += (curvature - self.max_curvature_) ** 2
        return cost

    # 计算平滑损失
    def __calcSmoothCost(self, path):
        cost = 0.0
        for i in range(1, len(path) - 1):
            point_m1 = Tools.pointToArray(path[i - 1])
            point = Tools.pointToArray(path[i])
            point_p1 = Tools.pointToArray(path[i + 1])
            delta_x = point - point_m1
            delta_px = point_p1 - point
            cost += np.dot((delta_px - delta_x).T, delta_px - delta_x)
        return cost

    # # 另一种曲率梯度计算方法用于验证曲率梯度计算
    # def __calcCurvatureItemCompare(self, path):
    #     eta = 0.001
    #     gradient = np.zeros((len(path), 2))
    #     for i in range(1, len(path) - 1):
    #         # 首先计算x方向的梯度
    #         changed_path1 = copy.deepcopy(path)
    #         changed_path2 = copy.deepcopy(path)
    #         changed_path1[i].x_ += eta
    #         changed_path2[i].x_ -= eta
    #         cost1 = self.__calcCurvatureCost(changed_path1)
    #         cost2 = self.__calcCurvatureCost(changed_path2)
    #         gradient[i][0] = (cost1 - cost2) / (2.0 * eta)
    #         # 再计算y方向的梯度
    #         changed_path1 = copy.deepcopy(path)
    #         changed_path2 = copy.deepcopy(path)
    #         changed_path1[i].y_ += eta
    #         changed_path2[i].y_ -= eta
    #         cost1 = self.__calcCurvatureCost(changed_path1)
    #         cost2 = self.__calcCurvatureCost(changed_path2)
    #         gradient[i][1] = (cost1 - cost2) / (2.0 * eta)
    #     return gradient

# 主函数
def main():
    # 设定起点和终点
    start = Point(10.0, 10.0)
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
    # 构建A星规划器
    astar_planner = AstarPlanner(start, goal, obstacles, config)
    planned_path = astar_planner.planning()
    # 可视化规划路径
    path_x, path_y = Tools.departPoints(planned_path)
    # 对路径进行平滑
    smoother = Smoother()
    if len(planned_path) > 5:
        smoothed_path = smoother.smoothing(planned_path, obstacles)
        # 可视化平滑路径
        spath_x, spath_y = Tools.departPoints(smoothed_path)
        plt.cla()
        plt.plot(spath_x, spath_y, "-g")
    plt.plot(obstalces_x, obstacles_y, ".k")
    plt.plot(start.x_, start.y_, "og")
    plt.plot(goal.x_, goal.y_, "xb")
    plt.grid(True)
    plt.axis("equal")
    plt.plot(path_x, path_y, "-r")
    plt.show()

if __name__ == "__main__":
    main()
