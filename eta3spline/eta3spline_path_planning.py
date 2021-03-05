#! /usr/bin/python3
# -*- coding: utf-8 -*-

"""

eta^3 spline path planning
author: flztiii

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# eta spline set类（由多条收尾相连的eta spline构成一条长路径）
class EtaSplineSet(object):
    def __init__(self, spline_segments):
        assert isinstance(spline_segments, list)
        assert len(spline_segments) > 0 and isinstance(spline_segments[0], EtaSpline)
        self.spline_segments_ = spline_segments
        self.n_ = len(spline_segments)
    
    # 计算特定参数对应的曲线上的坐标
    def calcPosition(self, u):
        assert u >= 0 and u <= self.n_
        # 首先判断u属于哪一段
        index = 0
        for i in range(0, self.n_):
            if i < self.n_ - 1:
                if u < i + 1:
                    index = i
                    break
            else:
                index = i
        u = u - index
        return self.spline_segments_[index].calcPosition(u)

# eta spline类
class EtaSpline(object):
    def __init__(self, start_pos, end_pos, eta):
        # 初始化条件
        self.start_pos_ = start_pos
        self.end_pos_ = end_pos
        self.eta_ = eta
        # 计算多项式参数
        self.__generateCoefficient()
        # 计算路程总长度
        self.total_distance_ = self.funcS(1)
    
    # 构造eta spline曲线七次多项式参数
    def __generateCoefficient(self):
        # 初始化参数
        coefficient = np.zeros((2, 8))
        # 根据公式计算多项式的参数
        coefficient[0, 0] = self.start_pos_[0]
        coefficient[0, 1] = self.eta_[0] * np.cos(self.start_pos_[2])
        coefficient[0, 2] = 0.5 * self.eta_[2] * np.cos(self.start_pos_[2]) - 0.5 * self.eta_[0] ** 2 * self.start_pos_[3] * np.sin(self.start_pos_[2])
        coefficient[0, 3] = self.eta_[4] * np.cos(self.start_pos_[2]) / 6.0 - (self.eta_[0] ** 3 * self.start_pos_[4] + 3 * self.eta_[0] * self.eta_[2] * self.start_pos_[3]) * np.sin(self.start_pos_[2]) / 6.0
        coefficient[0, 4] = 35.0 * (self.end_pos_[0] - self.start_pos_[0]) - (20 * self.eta_[0] + 5 * self.eta_[2] + 2.0/3.0 * self.eta_[4]) * np.cos(self.start_pos_[2]) + (5 * self.eta_[0] ** 2 * self.start_pos_[3] + 2.0/3.0 * self.eta_[0] ** 3 * self.start_pos_[4] + 2 * self.eta_[0] * self.eta_[2] * self.start_pos_[3]) * np.sin(self.start_pos_[2]) - (15 * self.eta_[1] - 5.0/2.0 * self.eta_[3] + 1.0/6.0 * self.eta_[5]) * np.cos(self.end_pos_[2]) - (5.0/2.0 * self.eta_[1] ** 2 * self.end_pos_[3] - 1.0/6.0 * self.eta_[1] ** 3 * self.end_pos_[4] - 0.5 * self.eta_[1] * self.eta_[3] * self.end_pos_[3]) * np.sin(self.end_pos_[2])
        coefficient[0, 5] = -84.0 * (self.end_pos_[0] - self.start_pos_[0]) + (45 * self.eta_[0] + 10 * self.eta_[2] + self.eta_[4]) * np.cos(self.start_pos_[2]) - (10 * self.eta_[0] ** 2 * self.start_pos_[3] + self.eta_[0] ** 3 * self.start_pos_[4] + 3 * self.eta_[0] * self.eta_[2] * self.start_pos_[3]) * np.sin(self.start_pos_[2]) + (39 * self.eta_[1] - 7 * self.eta_[3] + 0.5 * self.eta_[5]) * np.cos(self.end_pos_[2]) + (7 * self.eta_[1] ** 2 * self.end_pos_[3] - 1.0/2.0 * self.eta_[1] ** 3 * self.end_pos_[4] - 3.0/2.0 * self.eta_[1] * self.eta_[3] * self.end_pos_[3]) * np.sin(self.end_pos_[2])
        coefficient[0, 6] = 70.0 * (self.end_pos_[0] - self.start_pos_[0]) - (36 * self.eta_[0] + 15.0/2.0 * self.eta_[2] + 2.0/3.0 * self.eta_[4]) * np.cos(self.start_pos_[2]) + (15.0/2.0 * self.eta_[0] ** 2 * self.start_pos_[3] + 2.0/3.0 * self.eta_[0] ** 3 * self.start_pos_[4] + 2 * self.eta_[0] * self.eta_[2] * self.start_pos_[3]) * np.sin(self.start_pos_[2]) - (34 * self.eta_[1] - 13.0/2.0 * self.eta_[3] + 1.0/2.0 * self.eta_[5]) * np.cos(self.end_pos_[2]) - (13.0/2.0 * self.eta_[1] ** 2 * self.end_pos_[3] - 1.0/2.0 * self.eta_[1] ** 3 * self.end_pos_[4] - 3.0/2.0 * self.eta_[1] * self.eta_[3] * self.end_pos_[3]) * np.sin(self.end_pos_[2])
        coefficient[0, 7] = -20.0 * (self.end_pos_[0] - self.start_pos_[0]) + (10 * self.eta_[0] + 2 * self.eta_[2] + 1.0/6.0 * self.eta_[4]) * np.cos(self.start_pos_[2]) - (2 * self.eta_[0] ** 2 * self.start_pos_[3] + 1.0/6.0 * self.eta_[0] ** 3 * self.start_pos_[4] + 1.0/2.0 * self.eta_[0] * self.eta_[2] * self.start_pos_[3]) * np.sin(self.start_pos_[2]) + (10 * self.eta_[1] - 2 * self.eta_[3] + 1.0/6.0 * self.eta_[5]) * np.cos(self.end_pos_[2]) + (2 * self.eta_[1] ** 2 * self.end_pos_[3] - 1.0/6.0 * self.eta_[1] ** 3 * self.end_pos_[4] - 1.0/2.0 * self.eta_[1] * self.eta_[3] * self.end_pos_[3]) * np.sin(self.end_pos_[2])
        coefficient[1, 0] = self.start_pos_[1]
        coefficient[1, 1] = self.eta_[0] * np.sin(self.start_pos_[2])
        coefficient[1, 2] =  1.0/2.0 * self.eta_[2] * np.sin(self.start_pos_[2]) + 1.0/2.0 * self.eta_[0] ** 2 * self.start_pos_[3] * np.cos(self.start_pos_[2])
        coefficient[1, 3] = 1.0/6.0 * self.eta_[4] * np.sin(self.start_pos_[2]) + 1.0/6.0 * (self.eta_[0] ** 3 * self.start_pos_[4] + 3 * self.eta_[0] * self.eta_[2] * self.start_pos_[3]) * np.cos(self.start_pos_[2])
        coefficient[1, 4] = 35 * (self.end_pos_[1] - self.start_pos_[1]) - (20 * self.eta_[0] + 5 * self.eta_[2] + 2.0/3.0 * self.eta_[4]) * np.sin(self.start_pos_[2]) - (5 * self.eta_[0] ** 2 * self.start_pos_[3] + 2.0/3.0 * self.eta_[0] ** 3 * self.start_pos_[4] + 2 * self.eta_[0] * self.eta_[2] * self.start_pos_[3]) * np.cos(self.start_pos_[2]) - (15 * self.eta_[1] - 5.0/2.0 * self.eta_[3] + 1.0/6.0 * self.eta_[5]) * np.sin(self.end_pos_[2]) + (5.0/2.0 * self.eta_[1] ** 2 * self.end_pos_[3] - 1.0/6.0 * self.eta_[1] ** 3 * self.end_pos_[4] - 1.0/2.0 * self.eta_[1] * self.eta_[3] * self.end_pos_[3]) * np.cos(self.end_pos_[2])
        coefficient[1, 5] = -84 * (self.end_pos_[1] - self.start_pos_[1]) + (45 * self.eta_[0] + 10 * self.eta_[2] + self.eta_[4]) * np.sin(self.start_pos_[2]) + (10 * self.eta_[0] ** 2 * self.start_pos_[3] + self.eta_[0] ** 3 * self.start_pos_[4] + 3 * self.eta_[0] * self.eta_[2] * self.start_pos_[3]) * np.cos(self.start_pos_[2]) + (39 * self.eta_[1] - 7.0 * self.eta_[3] + 1.0/2.0 * self.eta_[5]) * np.sin(self.end_pos_[2]) - (7 * self.eta_[1] ** 2 * self.end_pos_[3] - 1.0/2.0 * self.eta_[1] ** 3 * self.end_pos_[4] - 3.0/2.0 * self.eta_[1] * self.eta_[3] * self.end_pos_[3]) * np.cos(self.end_pos_[2])
        coefficient[1, 6] = 70 * (self.end_pos_[1] - self.start_pos_[1]) - (36 * self.eta_[0] + 15.0/2.0 * self.eta_[2] + 2.0/3.0 * self.eta_[4]) * np.sin(self.start_pos_[2]) - (15.0/2.0 * self.eta_[0] ** 2 * self.start_pos_[3] + 2.0/3.0 * self.eta_[0] ** 3 * self.start_pos_[4] + 2 * self.eta_[0] * self.eta_[2] * self.start_pos_[3]) * np.cos(self.start_pos_[2]) - (34 * self.eta_[1] - 13.0/2.0 * self.eta_[3] + 1.0/2.0 * self.eta_[5]) * np.sin(self.end_pos_[2]) + (13.0/2.0 * self.eta_[1] ** 2 * self.end_pos_[3] - 1.0/2.0 * self.eta_[1] ** 3 * self.end_pos_[4] - 3.0/2.0 * self.eta_[1] * self.eta_[3] * self.end_pos_[3]) * np.cos(self.end_pos_[2])
        coefficient[1, 7] = -20 * (self.end_pos_[1] - self.start_pos_[1]) + (10 * self.eta_[0] + 2 * self.eta_[2] + 1.0/6.0 * self.eta_[4]) * np.sin(self.start_pos_[2]) + (2 * self.eta_[0] ** 2 * self.start_pos_[3] + 1.0/6.0 * self.eta_[0] ** 3 * self.start_pos_[4] + 1.0/2.0 * self.eta_[0] * self.eta_[2] * self.start_pos_[3]) * np.cos(self.start_pos_[2]) + (10 * self.eta_[1] - 2 * self.eta_[3] + 1.0/6.0 * self.eta_[5]) * np.sin(self.end_pos_[2]) - (2 * self.eta_[1] ** 2 * self.end_pos_[3] - 1.0/6.0 * self.eta_[1] ** 3 * self.end_pos_[4] - 1.0/2.0 * self.eta_[1] * self.eta_[3] * self.end_pos_[3]) * np.cos(self.end_pos_[2])
        self.coefficient_ = coefficient
    
    # 根据输入参数计算对应曲线上点的坐标
    def calcPosition(self, u):
        assert u >= 0 and u <= 1
        result = np.dot(self.coefficient_, np.array([1.0, u, u ** 2, u ** 3, u ** 4, u ** 5, u ** 6, u ** 7]).T)
        return result
    
    # 根据输入参数计算对应曲线上点的一阶导数
    def calcDerivative(self, u):
        assert u >= 0 and u <= 1
        result = np.dot(self.coefficient_[:,1:], np.array([1.0, 2 * u, 3 * u ** 2, 4 * u ** 3, 5 * u ** 4, 6 * u ** 5, 7 * u ** 6]).T)
        return result
    
    # 根据输入参数计算对应曲线上点的二阶导数
    def calc2Derivative(self, u):
        assert u >= 0 and u <= 1
        result = np.dot(self.coefficient_[:,2:], np.array([2, 6 * u, 12 * u ** 2, 20 * u ** 3, 30 * u ** 4, 42 * u ** 5]).T)
        return result
    
    # 根据输入参数计算s的导数
    def funcSDerivative(self, u):
        return max(np.linalg.norm(self.calcDerivative(u)), 1e-6)

    # 根据输入参数计算路程s
    def funcS(self, u):
        assert u >= 0 and u <= 1
        # 对ds进行积分
        s = integrate.quad(self.funcSDerivative, 0, u)[0]
        return s

# 主函数1，eta前两个参数发生变化
def main1():
    # 对于相同的边界条件使用不同的eta进行曲线的生成
    for i in range(0, 10):
        # 初始化eta
        eta = [i, i, 0, 0, 0, 0]  # eta为6维度自由变量
        # 初始化边界条件
        start_pos = [0, 0, 0, 0, 0]  # 起点位姿，形式为x坐标，y坐标，朝向yaw，曲率kappa，曲率导数
        end_pos = [4.0, 3.0, 0, 0, 0]  # 终点位姿，形式同上
        # 构造eta spline
        eta_spline = EtaSpline(start_pos, end_pos, eta)
        # 构造eta spline set
        eta_spline_segments = []
        eta_spline_segments.append(eta_spline)
        eta_spline_set = EtaSplineSet(eta_spline_segments)
        # 开始对eta spline进行采样，采样点数为1000个
        samples = np.linspace(0.0, len(eta_spline_segments), 1001)
        curve = []
        for sample in samples:
            sample_point = eta_spline_set.calcPosition(sample)
            curve.append(sample_point)
        curve = np.array(curve)
        # 进行可视化
        plt.plot(curve.T[0], curve.T[1])
        plt.axis("equal")
        plt.grid(True)
        plt.pause(1.0)
    plt.show()

# 主函数2，eta中间两个参数发生变化
def main2():
    # 对于相同的边界条件使用不同的eta进行曲线的生成
    for i in range(0, 10):
        # 初始化eta
        eta = [0, 0, (i - 5) * 20, (5 - i) * 20, 0, 0]  # eta为6维度自由变量
        # 初始化边界条件
        start_pos = [0, 0, 0, 0, 0]  # 起点位姿，形式为x坐标，y坐标，朝向yaw，曲率kappa，曲率导数
        end_pos = [4.0, 3.0, 0, 0, 0]  # 终点位姿，形式同上
        # 构造eta spline
        eta_spline = EtaSpline(start_pos, end_pos, eta)
        # 构造eta spline set
        eta_spline_segments = []
        eta_spline_segments.append(eta_spline)
        eta_spline_set = EtaSplineSet(eta_spline_segments)
        # 开始对eta spline进行采样，采样点数为1000个
        samples = np.linspace(0.0, len(eta_spline_segments), 1001)
        curve = []
        for sample in samples:
            sample_point = eta_spline_set.calcPosition(sample)
            curve.append(sample_point)
        curve = np.array(curve)
        # 进行可视化
        plt.plot(curve.T[0], curve.T[1])
        plt.axis("equal")
        plt.grid(True)
        plt.pause(1.0)
    plt.show()

# 主函数3，论文中曲线给出的参数
def main3():
    eta_spline_segments = []
    # 第一段参数, lane-change curve
    # 初始化eta
    eta = [4.27, 4.27, 0, 0, 0, 0]  # eta为6维度自由变量
    # 初始化边界条件
    start_pos = [0, 0, 0, 0, 0]  # 起点位姿，形式为x坐标，y坐标，朝向yaw，曲率kappa，曲率导数
    end_pos = [4.0, 1.5, 0, 0, 0]  # 终点位姿，形式同上
    # 构造eta spline
    eta_spline = EtaSpline(start_pos, end_pos, eta)
    eta_spline_segments.append(eta_spline)

    # 第二段参数, line segment
    # 初始化eta
    eta = [0, 0, 0, 0, 0, 0]  # eta为6维度自由变量
    # 初始化边界条件
    start_pos = [4, 1.5, 0, 0, 0]  # 起点位姿，形式为x坐标，y坐标，朝向yaw，曲率kappa，曲率导数
    end_pos = [5.5, 1.5, 0, 0, 0]  # 终点位姿，形式同上
    # 构造eta spline
    eta_spline = EtaSpline(start_pos, end_pos, eta)
    eta_spline_segments.append(eta_spline)

    # 第三段参数, cubic spiral
    # 初始化eta
    eta = [1.88, 1.88, 0, 0, 0, 0]  # eta为6维度自由变量
    # 初始化边界条件
    start_pos = [5.5, 1.5, 0, 0, 0]  # 起点位姿，形式为x坐标，y坐标，朝向yaw，曲率kappa，曲率导数
    end_pos = [7.4377, 1.8235, 0.6667, 1, 1]  # 终点位姿，形式同上
    # 构造eta spline
    eta_spline = EtaSpline(start_pos, end_pos, eta)
    eta_spline_segments.append(eta_spline)

    # 第四段参数, generic twirl arc
    # 初始化eta
    eta = [7, 10, 10, -10, 4, 4]  # eta为6维度自由变量
    # 初始化边界条件
    start_pos = [7.4377, 1.8235, 0.6667, 1, 1]  # 起点位姿，形式为x坐标，y坐标，朝向yaw，曲率kappa，曲率导数
    end_pos = [7.8, 4.3, 1.8, 0.5, 0]  # 终点位姿，形式同上
    # 构造eta spline
    eta_spline = EtaSpline(start_pos, end_pos, eta)
    eta_spline_segments.append(eta_spline)

    # 第五段参数, circular arc
    # 初始化eta
    eta = [2.98, 2.98, 0, 0, 0, 0]  # eta为6维度自由变量
    # 初始化边界条件
    start_pos = [7.8, 4.3, 1.8, 0.5, 0]  # 起点位姿，形式为x坐标，y坐标，朝向yaw，曲率kappa，曲率导数
    end_pos = [5.4581, 5.8064, 3.3416, 0.5, 0]  # 终点位姿，形式同上
    # 构造eta spline
    eta_spline = EtaSpline(start_pos, end_pos, eta)
    eta_spline_segments.append(eta_spline)

    eta_spline_set = EtaSplineSet(eta_spline_segments)
    # 开始对eta spline进行采样，采样点数为1000个
    samples = np.linspace(0.0, len(eta_spline_segments), 1001)
    curve = []
    for sample in samples:
        sample_point = eta_spline_set.calcPosition(sample)
        curve.append(sample_point)
    curve = np.array(curve)
    # 进行可视化
    plt.plot(curve.T[0], curve.T[1])
    plt.axis("equal")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # main1()
    # main2()
    main3()