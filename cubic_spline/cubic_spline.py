#! /usr/bin/python3
# -*- coding:utf-8 -*-

"""

cubic spline planning
author: flztiii

"""

import numpy as np
import math
import matplotlib.pyplot as plt
import bisect

# 构建三次样条曲线
class Spline:
    def __init__(self, x, y):
        assert len(x) == len(y)
        # 输入散点
        self.x_ = x
        self.y_ = y
        # 散点数量
        self.number_ = len(x)
        print("len", self.number_)
        # 最大下标(也是方程式的个数，参数个数)
        self.n_ = len(x) - 1
        # 中间参数
        self.h_ = np.diff(x)
        # 未知参数
        self.a_, self.b_, self.c_, self.d_ = [], [], [], []
        # 开始计算参数a
        self.a_ = y
        A = self.__getA()
        B = self.__getB()
        # 求解中间参量m
        m = np.linalg.solve(A, B)
        # 计算参数c
        self.c_ = m * 0.5
        # 计算参数d和b
        self.d_ = np.zeros(self.n_)
        self.b_ = np.zeros(self.n_)
        for i in range(0, len(self.d_)):
            self.d_[i] = (m[i + 1] - m[i]) / (6.0 * self.h_[i])
            self.b_[i] = (self.y_[i + 1] - self.y_[i]) / self.h_[i] - 0.5 * self.h_[i] * m[i] - self.h_[i] * (m[i + 1] - m[i]) / 6.0

    # 获得矩阵A
    def __getA(self):
        # 初始化矩阵A
        A = np.zeros((self.number_, self.number_))
        A[0, 0] = 1.0
        for i in range(1, self.n_):
            A[i, i] = 2 * (self.h_[i - 1] + self.h_[i])
            A[i - 1, i] = self.h_[i - 1]
            A[i, i - 1] = self.h_[i - 1]
        A[0, 1] = 0.0
        A[self.n_ - 1, self.n_] = self.h_[self.n_ - 1]
        A[self.n_, self.n_] = 1.0
        return A
    
    # 获得向量B，自由边界条件
    def __getB(self):
        B = np.zeros(self.number_)
        for i in range(1, self.n_):
            B[i] = 6.0 * ((self.y_[i + 1] - self.y_[i]) / self.h_[i] - (self.y_[i] - self.y_[i - 1]) / self.h_[i - 1])
        return B
    
    # 计算对应函数段
    def __getSegment(self, sample):
        return bisect.bisect(self.x_, sample) - 1
    
    # 计算位置
    def calc(self, sample):
        if sample < self.x_[0] or sample > self.x_[-1]:
            return None
        # 首先得到对应函数段
        index = self.__getSegment(sample)
        dx = sample - self.x_[index]
        return self.a_[index] + self.b_[index] * dx + self.c_[index] * dx ** 2 + self.d_[index] * dx ** 3
    
    # 计算一阶导数
    def calcd(self, sample):
        if sample < self.x_[0] or sample > self.x_[-1]:
            return None
        # 首先得到对应函数段
        index = self.__getSegment(sample)
        dx = sample - self.x_[index]
        return self.b_[index] + 2.0 * self.c_[index] * dx + 3.0 * self.d_[index] * dx ** 2

    # 计算二阶导数
    def calcdd(self, sample):
        if sample < self.x_[0] or sample > self.x_[-1]:
            return None
        # 首先得到对应函数段
        index = self.__getSegment(sample)
        dx = sample - self.x_[index]
        return 2.0 * self.c_[index] + 6.0 * self.d_[index] * dx

# 构建2d三次样条曲线
class CubicSpline2D:
    def __init__(self, x, y):
        assert len(x) == len(y)
        self.x_ = x
        self.y_ = y
        # 计算路程
        dx = np.diff(x)
        dy = np.diff(y)
        ds = [math.sqrt(idx ** 2 + idy ** 2) for (idx, idy) in zip(dx, dy)]
        self.s_ = [0.0]
        self.s_.extend(np.cumsum(ds))
        # 开始构建曲线
        self.__generateSpline()
    
    # 构建曲线
    def __generateSpline(self):
        self.spline_x_ = Spline(self.s_, self.x_)
        self.spline_y_ = Spline(self.s_, self.y_)
    
    # 计算位置
    def calcPosition(self, samples):
        x, y = [], []
        for sample in samples:
            if not self.spline_x_.calc(sample) is None:
                x.append(self.spline_x_.calc(sample))
                y.append(self.spline_y_.calc(sample))
        return x, y
    
    # 计算朝向
    def calcYaw(self, samples):
        yaws = []
        for sample in samples:
            dx = self.spline_x_.calcd(sample)
            dy = self.spline_y_.calcd(sample)
            if not dx is None:
                yaws.append(math.atan2(dy, dx))
        return yaws

    # 计算曲率
    def calcKappa(self, samples):
        kappas = []
        for sample in samples:
            dx = self.spline_x_.calcd(sample)
            dy = self.spline_y_.calcd(sample)
            ddx = self.spline_x_.calcdd(sample)
            ddy = self.spline_y_.calcdd(sample)
            if not dx is None:
                kappa = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))
                kappas.append(kappa)
        return kappas

# 主函数
def main():
    # 初始化散点
    x = [-2.5, 0.0, 2.5, 5.0, 7.5, 3.0, -1.0]
    y = [0.7, -6, 5, 6.5, 0.0, 5.0, -2.0]
    # 初始化采样间隔
    gap = 0.1

    # 构建2d三次样条曲线
    cubic_spline = CubicSpline2D(x, y)
    # 对2d三次样条曲线进行采样
    sample_s = np.arange(0.0, cubic_spline.s_[-1], gap)
    point_x, point_y = cubic_spline.calcPosition(sample_s)
    point_yaw = cubic_spline.calcYaw(sample_s)
    point_kappa = cubic_spline.calcKappa(sample_s)
    # 进行可视化
    plt.subplots(1)
    plt.plot(x, y, "xb", label="input")
    plt.plot(point_x, point_y, "-r", label="spline")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.legend()

    plt.subplots(1)
    plt.plot(sample_s, [np.rad2deg(iyaw) for iyaw in point_yaw], "-r", label="yaw")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("yaw angle[deg]")

    plt.subplots(1)
    plt.plot(sample_s, point_kappa, "-r", label="curvature")
    plt.grid(True)
    plt.legend()
    plt.xlabel("line length[m]")
    plt.ylabel("curvature [1/m]")

    plt.show()

# 测试函数
def test():
    x = [1., 2., 3., 4., 5., 6., 7.]
    y = [0.7, -6, 5, 6.5, 0.0, 5.0, -2.0]
    spline = Spline(x, y)
    value = spline.calc(2.9)
    derivative = spline.calcd(2.9)
    print(value, derivative)

if __name__ == "__main__":
    main()