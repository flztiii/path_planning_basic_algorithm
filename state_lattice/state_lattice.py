#! /usr/bin/python3
# -*- coding: utf-8 -*-

"""

State lattice method
author: flztiii

"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

# 将model predictive trajectory generation的工作目录加入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/model_predictive_trajectory_generation/")

import model_predictive_trajectory
import model

# 在查找表中找到最近的对应参数
def selectNearestTable(target, lookup_table):
    min_dis = float("inf")
    min_index = -1
    for i in range(len(lookup_table)):
        table = lookup_table[i]
        x_gap = table[0] - target.x_
        y_gap = table[1] - target.y_
        yaw_gap = table[2] - target.yaw_
        dis = math.sqrt(x_gap ** 2 + y_gap ** 2 + yaw_gap ** 2)
        if dis < min_dis:
            min_dis = dis
            min_index = i
    return lookup_table[min_index]
    

# 进行状态空间采样
def calcUniformSampledState(d, da_min, da_max, n_xy, ha_min, ha_max, n_h):
    # 采样终点状态
    targets = []
    for i in range(n_xy):
        # 起点到终点连线夹角
        da = da_min + i * (da_max - da_min) / (n_xy - 1)
        for j in range(n_h):
            # 终点车辆朝向角
            ha = ha_min + j * (ha_max - ha_min) / (n_h - 1) + da
            # 得到终点采样
            target = model.State(d * math.cos(da), d* math.sin(da), ha)
            targets.append(target)
    return targets

# 进行状态空间采样
def calcLaneSampledState(l_lateral, l_longitude, l_heading, l_width, vehicle_width, n_smapling):
    # 首先将车辆与道路路点之间的关系转化为路点的世界坐标系
    l_xc = l_longitude * math.cos(l_heading) + l_lateral * math.sin(l_heading)
    l_yc = l_longitude * math.sin(l_heading) + l_lateral * math.cos(l_heading)
    # 计算采样偏移
    offset = (l_width - vehicle_width) / (n_smapling - 1)
    # 得到采样状态
    targets = []
    for i in range(n_smapling):
        x = l_xc - (- 0.5 * (l_width - vehicle_width) + i * offset) * math.sin(l_heading)
        y = l_yc + (- 0.5 * (l_width - vehicle_width) + i * offset) * math.cos(l_heading)
        print(x, y)
        target = model.State(x, y, l_heading)
        targets.append(target)
    return targets

# 将终点转化为对应路径生成参数
def stateToParam(targets, k0):
    # 输入参数targets为State类型列表，k0为初始方向盘转角
    # 第一步加载查找表
    lookup_table = np.array(pd.read_csv(os.path.dirname(os.path.abspath(__file__))+"/lookuptable.csv"))
    # 第二步找出每一个终点对应的路径生成参数
    ps = []
    for target in targets:
        # 计算查找表中与当前终点最近的点
        selected_table = selectNearestTable(target, lookup_table)
        # 构造初始参数
        init_p = np.array([math.sqrt(target.x_ ** 2 + target.y_ ** 2), selected_table[4], selected_table[5]]).reshape((3, 1))
        # 进行model predictive得到优化参数
        x, y, yaw, p = model_predictive_trajectory.optimizeTrajectory(target, init_p, k0)
        # 将优化后路径对应的参数p保存
        if not x is None:
            ps.append(p)
    return ps

# 均匀状态空间采样
def uniformStateSpaceSampling():
    # 首先定义初始参数
    dis = 20.0  #  从起点到终点的距离
    dis_min_angle = np.deg2rad(-45)  # 起点到终点连线最小角度
    dis_max_angle = np.deg2rad(45)  # 起点到终点连线最大角度
    head_min_angle = np.deg2rad(-45)  # 终点最小朝向角
    head_max_angle = np.deg2rad(45)  # 终点最大朝向角
    n_xy_sample = 5  # 位置采样个数
    n_head_sample = 3  # 角度采样个数
    # 限制参数
    k0 = 0.0  # 初始方向盘转角
    # 进行终点采样，得到不同采样终点对应的路径生成参数
    assert(n_xy_sample > 1 and n_head_sample > 1)
    sampled_states = calcUniformSampledState(dis, dis_min_angle, dis_max_angle, n_xy_sample, head_min_angle, head_max_angle, n_head_sample)
    # 计算终点对应参数
    params = stateToParam(sampled_states, k0)
    # 用每一组参数生成路径，并进行可视化
    plt.clf()
    for param in params:
        x, y, yaw = model.generateTrajectory(param[0, 0], k0, param[1, 0], param[2, 0])
        plt.plot(x, y, '-r')
    plt.grid(True)
    plt.axis("equal")
    plt.show()

# 道路约束采样
def laneStateSpaceSampling():
    # 初始参数
    l_lateral = 10.0  # 起点在道路的横向偏移
    l_longitude = 6.0  # 起点在道路的纵向偏移
    l_heading = np.deg2rad(45)  # 道路朝向
    l_width = 3.0  # 道路宽度
    vehicle_width = 1.0  # 车辆宽度
    n_smapling = 5  # 采样个数

    # 车辆初始方向盘转角
    k0 = 0.0

    # 得到状态采样
    assert(n_smapling > 1)
    targets = calcLaneSampledState(l_lateral, l_longitude, l_heading, l_width, vehicle_width, n_smapling)
    # 将采样状态转化为路径生成参数
    params = stateToParam(targets, k0)
    # 根据参数生成路径
    plt.clf()
    for param in params:
        x, y, yaw = model.generateTrajectory(param[0, 0], k0, param[1, 0], param[2, 0])
        plt.plot(x, y, '-r')
    # 可视化
    plt.grid(True)
    plt.axis("equal")
    plt.show()


# 主函数
def main():
    # 均匀状态空间采样
    uniformStateSpaceSampling()
    # 目标点朝向密集采样
    # 道路约束采样
    laneStateSpaceSampling()

if __name__ == "__main__":
    main()