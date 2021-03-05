#! /usr/bin/python3
# -*- coding: utf-8 -*-

"""

Frenet optimal trajectory generation algrithom 
author: flztiii

"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/cubic_spline")
import cubic_spline

# 全局变量
ROAD_WIDTH = 7.0  # 道路的最大宽度
WIDTH_GAP = 1.0  # 道路的横向采样间隔
MIN_T = 4.0  # 最短时间开销
MAX_T = 5.0  # 最长时间开销
T_GAP = 0.1  # 时间采样间隔
ESP = 1e-6  # 精度要求
MAX_VELOCITY = 50.0 / 3.6  # 最大速度
TARGET_VELOCITY = 30.0 / 3.6  # 期望速度
VELOCITY_GAP = 5.0 / 3.6  # 速度采样间隔
VELOCITY_SAMPLE_N = 1  # 速度采样个数
MAX_KAPPA = 1.0  # 最大曲率
MAX_ACCELERATION = 2.0  # 最大加速度
ROBOT_SIZE = 2.0  # 机器人的半径
AREA = 20.0  # 可视化窗口大小

# 评分加权参数
KJ = 0.1
KT = 0.1
KD = 1.0
KLON = 1.0
KLAT = 1.0

# frenet轨迹对象
class FrenetPath(object):
    def __init__(self):
        self.t_ = []  # 时间采样

        # frenet系数据
        self.d_ = []  # 横向偏移
        self.d_derivative_ = []  # 横向偏移一阶导数
        self.d_2derivative_ = []  # 横向偏移二阶导数
        self.d_3derivative_ = []  # 横向偏移三阶导数
        self.s_ = []  # 纵向偏移
        self.s_derivative_ = []  # 纵向偏移一阶导数
        self.s_2derivative_ = []  # 纵向偏移二阶导数
        self.s_3derivative_ = []  # 纵向偏移三阶导数

        # 评分
        self.cost_ = 0.0

        # 世界系数据
        self.x_ = []  # x坐标
        self.y_ = []  # y坐标
        self.yaw_ = []  # 朝向角
        self.kappa_ = []  # 曲率
        self.ds_ = []  # 两点之间的距离

# 五次多项式
class QuinticPolynomial:
    def __init__(self, params, T):
        assert len(params) == 6
        self.params_ = params
        self.T_ = T
        # 生成多项式系数计算
        self.__coefficientGeneration()
        
    # 生成多项式系数计算
    def __coefficientGeneration(self):
        a0 = self.params_[0]
        a1 = self.params_[1]
        a2 = self.params_[2] * 0.5
        A = np.array([[self.T_ ** 3, self.T_ ** 4, self.T_ ** 5], [3. * self.T_ ** 2, 4. * self.T_ ** 3, 5. * self.T_ ** 4], [6. * self.T_, 12. * self.T_ ** 2, 20. * self.T_ ** 3]])
        b = np.array([self.params_[3] - a0 - a1 * self.T_ - a2 * self.T_ ** 2, self.params_[4] - a1 - 2. * a2 * self.T_, self.params_[5] - 2. * a2]).T
        x = np.linalg.solve(A, b)
        a3 = x[0]
        a4 = x[1]
        a5 = x[2]
        self.coefficient_ = np.array([a0, a1, a2, a3, a4, a5])

    # 给定输入时间，计算对应值
    def calcValue(self, t):
        return np.dot(self.coefficient_, np.array([1., t, t ** 2, t ** 3, t ** 4, t ** 5]).T)
    
    # 给定输入时间，计算导数
    def calcDerivation(self, t):
        return np.dot(self.coefficient_[1:], np.array([1., 2. * t, 3. * t ** 2, 4. * t ** 3, 5. * t ** 4]))
    
    # 给定输入时间，计算二阶导数
    def calc2Derivation(self, t):
        return np.dot(self.coefficient_[2:], np.array([2. , 6. * t, 12. * t ** 2, 20. * t ** 3]))
    
    # 给定输入时间，计算三阶导数
    def calc3Derivation(self, t):
        return np.dot(self.coefficient_[3:], np.array([6., 24. * t, 60. * t ** 2]))

# 四次多项式
class QuarticPolynomial:
    def __init__(self, xs, vs, acs, ve, ace, T):
        self.xs_ = xs
        self.vs_ = vs
        self.acs_ = acs
        self.ve_ = ve
        self.ace_ = ace
        self.T_ = T
        # 生成多项式系数计算
        self.__coefficientGeneration()

    # 生成多项式系数计算
    def __coefficientGeneration(self):
        a0 = self.xs_
        a1 = self.vs_
        a2 = self.acs_ * 0.5
        A = np.array([[3. * self.T_ ** 2, 4. * self.T_ ** 3], [6. * self.T_, 12. * self.T_ ** 2]])
        B = np.array([[self.ve_ - a1 - 2. * a2 * self.T_], [self.ace_ - 2. * a2]])
        x = np.linalg.solve(A, B)
        a3 = x[0, 0]
        a4 = x[1, 0]
        self.coefficient_ = np.array([a0, a1, a2, a3, a4])

    # 给定输入时间，计算对应值
    def calcValue(self, t):
        return np.dot(self.coefficient_, np.array([1., t, t ** 2, t ** 3, t ** 4]).T)
    
    # 给定输入时间，计算导数
    def calcDerivation(self, t):
        return np.dot(self.coefficient_[1:], np.array([1., 2. * t, 3. * t ** 2, 4. * t ** 3]))
    
    # 给定输入时间，计算二阶导数
    def calc2Derivation(self, t):
        return np.dot(self.coefficient_[2:], np.array([2. , 6. * t, 12. * t ** 2]))
    
    # 给定输入时间，计算三阶导数
    def calc3Derivation(self, t):
        return np.dot(self.coefficient_[3:], np.array([6., 24. * t]))

# frenet坐标架下的规划
def frenetOptimalTrajectoryPlanning(current_s, current_s_derivative, current_s_2derivative, current_d, current_d_derivative, curent_d_2derivative):
    # 生成轨迹组
    frenet_path_set = []
    # 首先是生横向偏移随时间变化，变化量为最终横向偏移和时间长度
    for df_offset in np.arange(- ROAD_WIDTH, ROAD_WIDTH, WIDTH_GAP):
        for T in np.arange(MIN_T, MAX_T + ESP, T_GAP):
            # 初始化轨迹对象
            frenet_path = FrenetPath()
            # 根据输入参数进行五次多项式拟合，输入参数[d0, d0_d, d0_dd, df, df_d, df_dd]
            quintic_polynomial = QuinticPolynomial([current_d, current_d_derivative, curent_d_2derivative, df_offset, 0.0, 0.0], T)
            # 对曲线进行采样
            sample_t = np.arange(0.0, T + ESP, T_GAP)
            frenet_path.t_ = [t for t in sample_t]
            frenet_path.d_ = [quintic_polynomial.calcValue(t) for t in frenet_path.t_]
            frenet_path.d_derivative_ = [quintic_polynomial.calcDerivation(t) for t in frenet_path.t_]
            frenet_path.d_2derivative_ = [quintic_polynomial.calc2Derivation(t) for t in frenet_path.t_]
            frenet_path.d_3derivative_ = [quintic_polynomial.calc3Derivation(t) for t in frenet_path.t_]

            # 开始生成纵向偏移随时间变化，采用的是速度保持模式，因此只需要用四次多项式
            for sf_derivative in np.arange(TARGET_VELOCITY - VELOCITY_GAP * VELOCITY_SAMPLE_N, TARGET_VELOCITY + VELOCITY_GAP * VELOCITY_SAMPLE_N + ESP, VELOCITY_GAP):
                # 赋值横向偏移数据
                frenet_path_copy = copy.deepcopy(frenet_path)
                # 根据输入参数进行四次多项式拟合（四次多项式不唯一，这里使用的一阶导数和二阶导数参数）
                quartic_polynomial = QuarticPolynomial(current_s, current_s_derivative, current_s_2derivative, sf_derivative, 0.0, T)
                frenet_path_copy.s_ = [quartic_polynomial.calcValue(t) for t in frenet_path.t_]
                frenet_path_copy.s_derivative_ = [quartic_polynomial.calcDerivation(t) for t in frenet_path.t_]
                frenet_path_copy.s_2derivative_ = [quartic_polynomial.calc2Derivation(t) for t in frenet_path.t_]
                frenet_path_copy.s_3derivative_ = [quartic_polynomial.calc3Derivation(t) for t in frenet_path.t_]

                # 得到轨迹在frenet系的全部信息
                # 开始给轨迹进行评分
                # 首先是横向评分，横向评分包括三部分
                # 第一部分是jerk
                lateral_jerk_cost = np.sum(np.power(frenet_path_copy.d_3derivative_, 2))
                # 第二部分是时间开销
                lateral_time_cost = T
                # 第三部分是最终横向偏移的平方
                lateral_offset_cost = frenet_path_copy.d_[-1] ** 2
                # 对以上三部分进行加权求和
                lateral_cost = KJ * lateral_jerk_cost + KT * lateral_time_cost + KD * lateral_offset_cost
                # 接下来是纵向评分
                # 纵向评分有三个部分
                # 第一部分是jerk
                longitudinal_jerk_cost = np.sum(np.power(frenet_path_copy.s_3derivative_, 2))
                # 第二部分是时间开销
                longitudinal_time_cost = T
                # 第三部分是最终速度偏差
                longitudinal_sderivative_offset = np.power(frenet_path_copy.s_derivative_[-1] - TARGET_VELOCITY, 2)
                # 对以上部分进行加权求和
                longitudinal_cost = KJ * longitudinal_jerk_cost + KT * longitudinal_time_cost + KD * longitudinal_sderivative_offset
                # 求解最终评分
                frenet_path_copy.cost_ = KLAT * lateral_cost + KLON * longitudinal_cost

                # 保存曲线到列表
                frenet_path_set.append(frenet_path_copy)

    return frenet_path_set

# frenet坐标系到世界坐标系的转化
def globalTrajectoryGeneration(frenet_path_set, reference_curve):
    global_path_set = []
    for frenet_path in frenet_path_set:
        global_path = copy.deepcopy(frenet_path)
        # 第一步补全位置信息x和y
        rfc_x, rfc_y = reference_curve.calcPosition(global_path.s_)
        rfc_yaw = reference_curve.calcYaw(global_path.s_)
        for i in range(0, len(rfc_x)):
            x = rfc_x[i] - np.sin(rfc_yaw[i]) * frenet_path.d_[i]
            y = rfc_y[i] + np.cos(rfc_yaw[i]) * frenet_path.d_[i]
            global_path.x_.append(x)
            global_path.y_.append(y)
        # 完成x,y信息的补全后通过计算的方法得到yaw和ds，计算方法为yaw=atan2(dy, dx)
        for i in range(0, len(global_path.x_) - 1):
            yaw = np.arctan2(global_path.y_[i + 1] - global_path.y_[i], global_path.x_[i + 1] - global_path.x_[i])
            global_path.yaw_.append(yaw)
            ds = np.sqrt((global_path.y_[i + 1] - global_path.y_[i]) ** 2 + (global_path.x_[i + 1] - global_path.x_[i]) ** 2)
            global_path.ds_.append(ds)
        global_path.yaw_.append(global_path.yaw_[-1])
        # 完成yaw的补全后通过计算得到kappa，计算方法为kappa=dyaw/ds
        for i in range(0, len(global_path.yaw_) - 1):
            kappa = (global_path.yaw_[i + 1] - global_path.yaw_[i]) / global_path.ds_[i]
            global_path.kappa_.append(kappa)
        global_path.kappa_.append(global_path.kappa_[-1])
        global_path_set.append(global_path)
    return global_path_set

# 判断是否发生碰撞
def isCollision(path, obstacles):
    for x, y in zip(path.x_, path.y_):
        for obstacle in obstacles:
            distance = np.sqrt((x - obstacle[0]) ** 2 + (y - obstacle[1]) ** 2)
            if distance <= ROBOT_SIZE:
                return True
    return False

# 判断路径组中路径是否可行
def checkPath(path_set, obstacles):
    checked_path = []
    # 判断路径是否可行存在四个方面
    for i in range(0, len(path_set)):
        # 第一个方面判断是否有大于最大速度的速度
        if max([np.abs(v) for v in path_set[i].s_derivative_]) > MAX_VELOCITY:
            continue
        # 第二个方面判断是否有大于最大曲率的曲率
        if max([np.abs(kappa) for kappa in path_set[i].kappa_]) > MAX_KAPPA:
            continue
        # 第三个方面判断是否有大于最大加速度的加速度
        if max([np.abs(acceleration) for acceleration in path_set[i].s_2derivative_]) > MAX_ACCELERATION:
            continue
        # 第四个方面，判断是否与障碍物存在碰撞
        if isCollision(path_set[i], obstacles):
            continue
        checked_path.append(path_set[i])
    return checked_path

# 主函数
def main():
    # 初始化参考路点
    wx = [0.0, 10.0, 20.5, 35.0, 70.5]
    wy = [0.0, -6.0, 5.0, 6.5, 0.0]

    # 初始化障碍物所在位置
    obstacles = np.array([[20.0, 10.0],
                   [30.0, 6.0],
                   [30.0, 8.0],
                   [35.0, 8.0],
                   [50.0, 3.0]
                   ])
    
    # 初始化车辆状态信息
    current_s = 0.0  # frenet坐标下的s轴
    current_s_derivative = 10.0 / 3.6  # frenet坐标下的s轴对时间的一阶导数
    current_s_2derivative = 0.0  # frenet坐标下的s轴对时间的二阶导数
    current_d = 2.0  # frenet坐标下的d轴
    current_d_derivative= 0.0  # frenet坐标下的d轴对时间的一阶导数
    curent_d_2derivative = 0.0  # frenet坐标下的d轴对时间的二阶阶导数

    # 接下来是规划的主体内容

    # 第一步生成参考路点路径曲线
    # 利用三次样条差值生成参考曲线
    reference_curve = cubic_spline.CubicSpline2D(wx, wy)
    reference_samples = np.arange(0.0, reference_curve.s_[-1], 0.1)
    reference_waypoints_x, reference_waypoints_y = reference_curve.calcPosition(reference_samples)

    # 记录当前点的朝向和曲率
    yaw_set, kappa_set, s_set = [], [], []

    # 每一次更新所花费时间
    gap = 1
    # 开始进行规划
    while True:
        # 原算法以10hz频率不断进行规划，此处以到达下一个点进行替代，每到规划的下一个点就进行重新规划
        # 进行frenet坐标架下的规划
        frenet_path_set = frenetOptimalTrajectoryPlanning(current_s, current_s_derivative, current_s_2derivative, current_d, current_d_derivative, curent_d_2derivative)

        # 将frenet坐标系转换到世界坐标系下
        global_path_set = globalTrajectoryGeneration(frenet_path_set, reference_curve)
        # 判断路径组中是否可行
        # print(len(global_path_set))
        checked_path_set = checkPath(global_path_set, obstacles)
        # print(len(checked_path_set))
        # 从其中选取一条代价最低的作为最终路径
        # min_cost = float("inf")
        final_path = min(checked_path_set, key=lambda p: p.cost_)

        # 新的下标
        current_index = min(gap, len(final_path.x_) - 1)
        # 得到新的起点
        current_s = final_path.s_[current_index]
        current_s_derivative = final_path.s_derivative_[current_index]
        current_s_2derivative = 0.0
        current_d = final_path.d_[current_index]
        current_d_derivative= final_path.d_derivative_[current_index]
        curent_d_2derivative = final_path.d_2derivative_[current_index]

        # 记录当前点的朝向和曲率
        for i in range(0, current_index):
            yaw_set.append(final_path.yaw_[i])
            kappa_set.append(final_path.kappa_[i])
            s_set.append(final_path.s_[i])

        # 判断是否已经到达终点范围
        if np.sqrt((final_path.x_[current_index] - wx[-1]) ** 2 + (final_path.y_[current_index] - wy[-1]) ** 2) < ROBOT_SIZE:
            print("goal reached")
            break

        # 进行可视化
        # 首先清空之前的可视化内容
        plt.cla()
        # 可视化参考道路线
        plt.plot(reference_waypoints_x, reference_waypoints_y)
        # 可视化障碍物点
        plt.plot(obstacles[:, 0], obstacles[:, 1], "xk")
        # 可视化当前位置
        plt.plot(final_path.x_[1], final_path.y_[1], "vc")
        # 可视化路径
        plt.plot(final_path.x_[1:], final_path.y_[1:], "-or")
        # 可视化窗口
        plt.xlim(final_path.x_[1] - AREA, final_path.x_[1] + AREA)
        plt.ylim(final_path.y_[1] - AREA, final_path.y_[1] + AREA)
        plt.title("v[km/h]:" + str(current_s_derivative * 3.6)[0:4])
        plt.grid(True)
        plt.pause(0.0001)
    plt.grid(True)
    plt.pause(0.0001)
    # 可视化朝向和曲率的变化
    # plt.subplots()
    # plt.grid(True)
    # plt.plot(s_set, yaw_set)
    plt.subplots()
    plt.title('hz is' + str(1.0/gap))
    plt.grid(True)
    plt.plot(s_set, kappa_set)
    plt.show()

if __name__ == "__main__":
    main()