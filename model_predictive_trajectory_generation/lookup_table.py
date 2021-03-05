#! /usr/bin/python3
# -*- coding: utf-8 -*-

"""

lookup table
author: flztiii

"""

import matplotlib.pyplot as plt
import numpy as np
import math
import model
import model_predictive_trajectory
import pandas as pd
import os

# 初始化查询表范围
def initSearchRange():
    # 初始化朝向范围
    max_yaw = np.deg2rad(-30.0)
    yaw_range = np.arange(-max_yaw, max_yaw, max_yaw)
    # 初始化xy范围
    x_range = np.arange(10.0, 30.0, 5.0)
    y_range = np.arange(0.0, 20.0, 2.0)
    # 得到范围
    states = []
    for yaw in yaw_range:
        for y in y_range:
            for x in x_range:
                states.append([x, y, yaw])
    return states

# 计算距离
def calcDistance(state, table):
    """
        :param state: 要计算距离的状态
        :param table: 查找表中元素
    """

    dx = state[0] - table[0]
    dy = state[1] - table[1]
    dyaw = state[2] - table[2]
    return math.sqrt(dx ** 2 + dy ** 2 + dyaw ** 2)

# 保存查找表
def saveLookupTable(fname, lookup_table):
    mt = np.array(lookup_table)
    # save csv
    df = pd.DataFrame()
    df["x"] = mt[:, 0]
    df["y"] = mt[:, 1]
    df["yaw"] = mt[:, 2]
    df["s"] = mt[:, 3]
    df["km"] = mt[:, 4]
    df["kf"] = mt[:, 5]

    df.to_csv(fname, index=None)

# 构建查找表
def generateLookupTable():
    # 第一步构造查找表初始值
    # 查找表中每一个元素包括x, y, yaw, s, km, kf
    k0 = 0.0
    lookup_table = [[1.0, .0, .0, 1.0, .0, .0]]
    # 初始化查找表范围
    states = initSearchRange()
    # 遍历查找表范围内的全部初始状态
    for state in states:
        print("state", state)
        # 找到当前查找表中离当前点最近的点
        min_dis = float("inf")
        min_index = -1
        for i in range(len(lookup_table)):
            dis = calcDistance(state, lookup_table[i])
            if dis <= min_dis:
                min_dis = dis
                min_index = i
        print("index", i, "min_dis", min_dis)
        table = lookup_table[min_index]
        print("bestp", table)
        # 得到路径优化初始条件
        init_p = np.array([math.sqrt(state[0] ** 2 + state[1] ** 2), table[4], table[5]]).reshape((3, 1))
        target = model.State(state[0], state[1], state[2])
        # 生成路径
        x, y, yaw, p = model_predictive_trajectory.optimizeTrajectory(target, init_p, k0)
        if not x is None:
            # 加入查找表
            lookup_table.append([x[-1], y[-1], yaw[-1], float(p[0]), float(p[1]), float(p[2])])

    # 完成查找表的生成，将表进行保存
    # saveLookupTable((os.getcwd() + "/" + __file__).replace(".py", ".csv"), lookup_table)
    # 查找表中参数生成路径的可视化
    for table in lookup_table:
        x, y, yaw = model.generateTrajectory(table[3], k0, table[4], table[5])
        plt.plot(x, y, "-r")
        x, y, yaw = model.generateTrajectory(table[3], k0, -table[4], -table[5])
        plt.plot(x, y, "-r")
    plt.grid(True)
    plt.axis("equal")
    plt.show()
# 主函数
def main():
    generateLookupTable()

if __name__ == "__main__":
    main()