#! /usr/bin/python3 
# -*- coding: utf-8 -*-

"""

model predictive trajectory 
author: flztiii

"""

import numpy as np
import model
import math
import matplotlib.pyplot as plt

# 优化参数全局变量
max_iter = 100  # 最大迭代次数
h = np.array([0.5, 0.02, 0.02]).T  # 采样距离参数
cost_th = 0.1  # 迭代完成阈值

# 角度转弧度
def degToRed(degree):
    return model.pi2Pi(degree / 180.0 * np.pi)

# 判断点与目标之间的差距
def calcDiff(x, y, yaw, target):
    x_diff = target.x_ - x
    y_diff = target.y_ - y
    yaw_diff = model.pi2Pi(target.yaw_ - yaw)
    return np.array([x_diff, y_diff, yaw_diff])

# 可视化箭头
def plotArrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              fc=fc, ec=ec, head_width=width, head_length=width)
    plt.plot(x, y)
    plt.plot(0, 0)

# 可视化路径
def plotTrajectory(xc, yc, target):
    plt.clf()
    plotArrow(target.x_, target.y_, target.yaw_)
    plt.plot(xc, yc, "-r")
    plt.axis("equal")
    plt.grid(True)
    plt.pause(1)

# 计算雅克比矩阵
def calcJaccobi(target, p, k0):
    # 雅克比矩阵为横轴为对变量不同维度的导数，此处利用离散化来计算每一个变量维度的导数
    # 首先是第一维度变量导数计算
    prev_x, prev_y, prev_yaw = model.generateFinalState(p[0,0] - h[0], k0, p[1, 0], p[2, 0])
    next_x, next_y, next_yaw = model.generateFinalState(p[0,0] + h[0], k0, p[1, 0], p[2, 0])
    prev_diff = calcDiff(prev_x, prev_y, prev_yaw, target)
    next_diff = calcDiff(next_x, next_y, next_yaw, target)
    d1 = ((next_diff - prev_diff) / (2.0 * h[0])).reshape((3, 1))
    # 是第二维度变量导数计算
    prev_x, prev_y, prev_yaw = model.generateFinalState(p[0,0], k0, p[1, 0] - h[1], p[2, 0])
    next_x, next_y, next_yaw = model.generateFinalState(p[0,0], k0, p[1, 0] + h[1], p[2, 0])
    prev_diff = calcDiff(prev_x, prev_y, prev_yaw, target)
    next_diff = calcDiff(next_x, next_y, next_yaw, target)
    d2 = ((next_diff - prev_diff) / (2.0 * h[1])).reshape((3, 1))
    # 是第三维度变量导数计算
    prev_x, prev_y, prev_yaw = model.generateFinalState(p[0,0], k0, p[1, 0], p[2, 0] - h[2])
    next_x, next_y, next_yaw = model.generateFinalState(p[0,0], k0, p[1, 0], p[2, 0] + h[2])
    prev_diff = calcDiff(prev_x, prev_y, prev_yaw, target)
    next_diff = calcDiff(next_x, next_y, next_yaw, target)
    d3 = ((next_diff - prev_diff) / (2.0 * h[2])).reshape((3, 1))
    # 得到雅克比矩阵
    jaccobi = np.hstack((d1, d2, d3))
    return jaccobi

# 计算学习率
def selectLearningRate(gradient, target, p, k0):
    # 初始化上下限
    min_rate = 1.0
    max_rate = 2.0
    rate_gap = 0.5
    # 初始化最小值
    min_cost = float("inf")
    final_rate = min_rate
    # 找出能使代价函数最小的学习率
    for rate in np.arange(min_rate, max_rate, rate_gap):
        tp = p + rate * gradient
        x, y, yaw = model.generateFinalState(tp[0, 0], k0, tp[1, 0], tp[2, 0])
        diff = calcDiff(x, y, yaw, target).reshape((3, 1))
        diff_norm = np.linalg.norm(diff)
        if diff_norm < min_cost and rate != 0.0:
            min_cost = diff_norm
            final_rate = rate
    return final_rate

# 优化路径生成
def optimizeTrajectory(target, p, k0):
    for i in range(0, max_iter):
        # print("interaction", i, "param", p)
        # if i == max_iter - 1:
            # print("reach max iteration number")
        # 首先利用当前参数进行路径生成
        x, y, yaw = model.generateTrajectory(p[0, 0], k0, p[1, 0], p[2, 0])
        # 判断路径最后一个点与目标点的差值
        target_diff = calcDiff(x[-1], y[-1], yaw[-1], target).reshape((3, 1))
        # 计算差值的欧式距离
        target_diff_norm = np.linalg.norm(target_diff)
        # 判断距离是否小于阈值，如果小于阈值则完成
        if target_diff_norm < cost_th:
            # 完成迭代
            print("iteration finished cost", target_diff_norm)
            break
        else:
            # 没有完成迭代
            # print("interation", i, "cost", target_diff_norm)
            # 计算二次范数的最速梯度
            # 首先计算雅克比矩阵
            jaccobi = calcJaccobi(target, p, k0)
            # 计算二次范数梯度方向
            try:
                inv_jaccobi = np.linalg.inv(jaccobi)
            except np.linalg.LinAlgError:
                # print("no jaccobi inv")
                x, y, yaw, p = None, None, None, None
                break
            # 得到梯度
            gradient = - inv_jaccobi @ target_diff
            # 计算学习率
            alpha = selectLearningRate(gradient, target, p, k0)
            # 更新参数
            p = p + alpha * gradient
            # 可视化当前轨迹
            plotTrajectory(x, y, target)
    return x, y, yaw, p

# 优化路径生成算法测试函数
def testOptimizeTrajectory():
    # 确定目标点
    target = model.State(x = 8.0, y = 3.0, yaw = degToRed(60))
    # 初始化参数
    p = np.array([6.0, 0.0, 0.0]).reshape((3,1))
    # 初始方向盘转角
    k0 = 0.0
    # 开始优化路径生成  
    x, y, yaw, p = optimizeTrajectory(target, p, k0)
    # 可视化最终轨迹
    plotTrajectory(x, y, target)
    plotArrow(target.x_, target.y_, target.yaw_)
    plt.axis("equal")
    plt.grid(True)
    plt.show()

# 主函数
def main():
    # print(__file__ + " start!!")
    testOptimizeTrajectory()

if __name__ == "__main__":
    main()