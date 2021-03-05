#! /usr/bin/python3
# -*- coding: utf-8 -*-

"""

bezier planning (三阶贝塞尔曲线规划)
author: flztiii

"""

import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.special

# 贝赛尔曲线类(三阶以上)
class Bezier:
    def __init__(self, control_points):
        self.control_points_ = control_points  # 控制点
        self.n_ = len(control_points) - 1  # 贝赛尔曲线的阶数
        if self.n_ > 2:
            self.__calcd()  # 计算贝赛尔曲线一阶导数的控制点
            self.__calcdd()  # 计算贝赛尔曲线二阶导数的控制点

    # 获取贝赛尔曲线特定位置
    def calcPosition(self, t):
        position = np.zeros(2)
        for i in range(0, self.n_ + 1):
            position += self.__calcBezierCoefficient(self.n_, i, t) * self.control_points_[i]
        return position
    
    # 获取贝赛尔曲线特定导数
    def calcd(self, t):
        d = np.zeros(2)
        for i in range(0, self.n_):
            d += self.__calcBezierCoefficient(self.n_ - 1, i, t) * self.control_points_d_[i]
        return d
    
    # 获取贝赛尔曲线特定二阶导数
    def calcdd(self, t):
        dd = np.zeros(2)
        for i in range(0, self.n_ - 1):
            dd += self.__calcBezierCoefficient(self.n_ - 2, i, t) * self.control_points_dd_[i]
        return dd
    
    # 计算对应的bezier系数
    def __calcBezierCoefficient(self, n, i, t):
        return scipy.special.comb(n, i) * (1 - t) ** (n - i) * t ** i
    
    # 计算贝赛尔曲线一阶导数的控制点
    def __calcd(self):
        self.control_points_d_ = []
        for i in range(0, self.n_):
            self.control_points_d_.append(self.n_ * (self.control_points_[i + 1] - self.control_points_[i]))
        self.control_points_d_ = np.array(self.control_points_d_)
    
    # 计算贝赛尔曲线二阶导数控制点
    def __calcdd(self):
        self.control_points_dd_ = []
        for i in range(0, self.n_ - 1):
            self.control_points_dd_.append((self.n_ - 1) * (self.control_points_d_[i + 1] - self.control_points_d_[i]))
        self.control_points_dd_ = np.array(self.control_points_dd_)

# 生成三阶贝赛尔曲线的四个控制点
def generateControlPoints(start_x, start_y, start_yaw, end_x, end_y, end_yaw, offset):
    # 起点到终点的距离
    distance = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
    # 起点
    control_points = [[start_x, start_y]]
    # 起点切线控制点
    control_points.append([start_x + np.cos(start_yaw) * distance / offset, start_y + np.sin(start_yaw) * distance / offset])
    # 终点切线控制点
    control_points.append([end_x - np.cos(end_yaw) * distance / offset, end_y - np.sin(end_yaw) * distance / offset])
    # 终点
    control_points.append([end_x, end_y])
    return np.array(control_points)

# 生成三阶贝赛尔曲线
def generateBezier(start_x, start_y, start_yaw, end_x, end_y, end_yaw, offset):
    # 第一步根据输入信息生成三阶贝赛尔曲线控制点，因此控制点的数量为四个
    control_points = generateControlPoints(start_x, start_y, start_yaw, end_x, end_y, end_yaw, offset)
    # 构造贝赛尔曲线
    bezier_curve = Bezier(control_points)

    return bezier_curve, control_points

# 计算曲率
def calcKappa(dx, dy, ddx, ddy):
    return (dx * ddy - dy * ddx) / math.pow(dx ** 2 + dy ** 2, 1.5)

# 可视化朝向
def plotArrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):  # pragma: no cover
    """Plot arrow."""
    if not isinstance(x, float):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * np.cos(yaw), length * np.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)

# 主函数
def main():
    # 初始化起点
    start_x = 10.0  # [m]
    start_y = 1.0  # [m]
    start_yaw = np.radians(180.0)  # [rad]

    end_x = -0.0  # [m]
    end_y = -3.0  # [m]
    end_yaw = np.radians(-45.0)  # [rad]
    offset = 3.0  # 控制点的等分数
    bezier_curve, control_points = generateBezier(start_x, start_y, start_yaw, end_x, end_y, end_yaw, offset)

    # 得到贝赛尔曲线的采样
    sample_number = 100
    samples = np.linspace(0, 1, sample_number)
    curve_points = []
    for sample in samples:
        curve_points.append(bezier_curve.calcPosition(sample))
    curve_points = np.array(curve_points)
    # 贝赛尔曲线特定点信息
    special_sample = 0.86
    # 计算此点的位置
    point = bezier_curve.calcPosition(special_sample)
    # 计算此点的朝向
    point_d = bezier_curve.calcd(special_sample)
    point_d_norm = point_d / np.linalg.norm(point_d, 2)
    # 计算此点的切向单位向量
    targent = np.array([point, point + point_d_norm])
    # 计算此点的法向单位向量
    norm = np.array([point, point + np.array([- point_d_norm[1], point_d_norm[0]])])
    # 计算此点的曲率半径
    point_dd = bezier_curve.calcdd(special_sample)
    point_kappa = calcKappa(point_d[0], point_d[1], point_dd[0], point_dd[1])
    radius = 1.0 / point_kappa
    # 计算圆心位置
    circular_center = point + radius * np.array([- point_d_norm[1], point_d_norm[0]])
    circle = plt.Circle(tuple(circular_center), radius, color=(0, 0.8, 0.8), fill=False, linewidth=1)

    # 可视化
    fig, ax = plt.subplots()
    # 可视化控制点
    ax.plot(control_points.T[0], control_points.T[1], '--o', label = 'control points')
    # 可视化贝赛尔曲线
    ax.plot(curve_points.T[0], curve_points.T[1], label = 'bezier curve')
    # 可视化曲率圆
    ax.add_artist(circle)
    # 可视化特定点的切向单位向量
    ax.plot(targent.T[0], targent.T[1], label = 'targent')
    # 可视化特定点的法向单位向量
    ax.plot(norm.T[0], norm.T[1], label = 'norm')
    # 可视化起点和终点朝向
    plotArrow(start_x, start_y, start_yaw)
    plotArrow(end_x, end_y, end_yaw)
    ax.legend()
    ax.axis("equal")
    ax.grid(True)
    plt.show()

if __name__ == "__main__":
    main()