#! /usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# 计算角度与2pi求余,计算出的角度在[-2pi,2pi]区间内
def mod2Pi(theta):
    return theta - 2.0 * np.pi * np.floor(theta / 2.0 / np.pi)

# 将[-2pi,2pi]转换到[-pi,pi]
def pi2Pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# 坐标转换，将原始坐标转换到coordinate坐标下
def transformCoordinate(coordinate, origin):
    new_yaw = origin[2] - coordinate[2]
    new_x = (origin[0] - coordinate[0]) * np.cos(coordinate[2]) + (origin[1] - coordinate[1]) * np.sin(coordinate[2])
    new_y = - (origin[0] - coordinate[0]) * np.sin(coordinate[2]) + (origin[1] - coordinate[1]) * np.cos(coordinate[2])
    return [new_x, new_y, new_yaw]

# 坐标恢复,将coordinate坐标转换到原始坐标
def retransfromCoordinate(coordinate, transformed):
    origin_yaw = pi2Pi(transformed[2] + coordinate[2])
    origin_x = coordinate[0] + transformed[0] * np.cos(coordinate[2]) - transformed[1] * np.sin(coordinate[2])
    origin_y = coordinate[1] + transformed[0] * np.sin(coordinate[2]) + transformed[1] * np.cos(coordinate[2])
    return [origin_x, origin_y, origin_yaw]

# 定义模式LSL
def LSL(alpha, d, beta):
    # 模式
    mode = ["L", "S", "L"]

    # 计算三角函数
    sa = np.sin(alpha)
    sb = np.sin(beta)
    ca = np.cos(beta)
    cb = np.cos(beta)
    cab = np.cos(alpha - beta)

    # 计算中间量，判断是否存在
    p_square = 2. + d ** 2 - 2. * cab + 2. * d * (sa - sb)
    if p_square < 0:
        return None, None, None, None

    # 计算t,p,q
    t = mod2Pi(-alpha + np.arctan((cb - ca) / (d + sa - sb)))
    p = np.sqrt(p_square)
    q = mod2Pi(beta - np.arctan((cb - ca) / (d + sa - sb)))

    return t, p, q, mode

# 定义模式RSR
def RSR(alpha, d, beta):
    # 模式
    mode = ["R", "S", "R"]

    # 计算三角函数
    sa = np.sin(alpha)
    sb = np.sin(beta)
    ca = np.cos(beta)
    cb = np.cos(beta)
    cab = np.cos(alpha - beta)

    # 计算中间量，判断是否存在
    p_square = 2. + d ** 2 - 2. * cab + 2. * d * (sb - sa)
    if p_square < 0:
        return None, None, None, None

    # 计算t,p,q
    t = mod2Pi(alpha - np.arctan((ca - cb) / (d - sa + sb)))
    p = np.sqrt(p_square)
    q = mod2Pi(-mod2Pi(beta) + np.arctan((ca - cb) / (d - sa + sb)))

    return t, p, q, mode

# 定义模式RSL
def RSL(alpha, d, beta):
    # 模式
    mode = ["R", "S", "L"]

    # 计算三角函数
    sa = np.sin(alpha)
    sb = np.sin(beta)
    ca = np.cos(beta)
    cb = np.cos(beta)
    cab = np.cos(alpha - beta)

    # 计算中间量，判断是否存在
    p_square = -2. + d ** 2 + 2. * cab + 2. * d * (sa + sb)
    if p_square < 0:
        return None, None, None, None

    # 计算t,p,q
    p = np.sqrt(p_square)
    t = mod2Pi(-alpha + np.arctan((- ca - cb) / (d + sa + sb)) - np.arctan(-2. / p)) 
    q = mod2Pi(-mod2Pi(beta) + np.arctan((- ca - cb) / (d + sa + sb)) - np.arctan(-2. / p))

    return t, p, q, mode

# 定义模式LSR
def LSR(alpha, d, beta):
    # 模式
    mode = ["L", "S", "R"]

    # 计算三角函数
    sa = np.sin(alpha)
    sb = np.sin(beta)
    ca = np.cos(beta)
    cb = np.cos(beta)
    cab = np.cos(alpha - beta)

    # 计算中间量，判断是否存在
    p_square = -2. + d ** 2 + 2. * cab - 2. * d * (sa + sb)
    if p_square < 0:
        return None, None, None, None

    # 计算t,p,q
    p = np.sqrt(p_square)
    t = mod2Pi(alpha - np.arctan((ca + cb) / (d - sa - sb)) + np.arctan(2. / p)) 
    q = mod2Pi(mod2Pi(beta) - np.arctan((ca + cb) / (d - sa - sb)) + np.arctan(2. / p))

    return t, p, q, mode

# 定义模式RLR
def RLR(alpha, d, beta):
    # 模式
    mode = ["R", "L", "R"]

    # 计算三角函数
    sa = np.sin(alpha)
    sb = np.sin(beta)
    ca = np.cos(beta)
    cb = np.cos(beta)
    cab = np.cos(alpha - beta)

    # 计算中间量，判断是否存在
    pc = (6. - d ** 2 + 2. * cab + 2. * d * (sa - sb)) / 8.
    if abs(pc) > 1:
        return None, None, None, None

    # 计算t,p,q
    p = np.arccos(pc)
    t = mod2Pi(alpha - np.arctan((ca - cb) / (d - sa + sb)) + p / 2.)
    q = mod2Pi(alpha - beta - t + p)

    return t, p, q, mode

# 定义模式LRL
def LRL(alpha, d, beta):
    # 模式
    mode = ["L", "R", "L"]

    # 计算三角函数
    sa = np.sin(alpha)
    sb = np.sin(beta)
    ca = np.cos(beta)
    cb = np.cos(beta)
    cab = np.cos(alpha - beta)

    # 计算中间量，判断是否存在
    pc = (6. - d ** 2 + 2. * cab + 2. * d * (sa - sb)) / 8.
    if abs(pc) > 1:
        return None, None, None, None

    # 计算t,p,q
    p = mod2Pi(np.arccos(pc))
    t = mod2Pi(-alpha + np.arctan((-ca + cb) / (d + sa - sb)) + p / 2.)
    q = mod2Pi(mod2Pi(beta) - alpha + 2. * p)

    return t, p, q, mode

# 原始dubins曲线规划
def rawPlanning(transformed_start_pose, transformed_end_pose, curvature):
    print(transformed_start_pose)
    print(transformed_end_pose)
    # 首先初始化dubins曲线生成的6种模式
    func_modes = [LSL, RSR, LSR, RSL, RLR, LRL]
    shortest_t, shortest_p, shortest_q, shortest_mode = 0.0, 0.0, 0.0, []
    # 分别调用不同模式计算各段长度t,p,q
    # 计算比较哪种模式下总长度最短，选中此模式
    D = transformed_end_pose[0]
    d = D * curvature
    min_cost = float("inf")
    for func_mode in func_modes:
        t, p, q, mode = func_mode(transformed_start_pose[2], d, transformed_end_pose[2])
        if t is None:
            continue
        L = t + p + q
        if L < min_cost:
            min_cost = L
            shortest_t = t
            shortest_p = p
            shortest_q = q
            shortest_mode = mode
    print(shortest_t, shortest_p, shortest_q, shortest_mode)
    # 根据得到的t,p,q进行点采样
    x, y, yaw = generateCurve(transformed_start_pose, [shortest_t, shortest_p, shortest_q], shortest_mode, curvature)
    return x, y, yaw


# 根据t,p,q进行点采样
def generateCurve(start_pose, lengths, mode, curvature, gap = np.deg2rad(5.0)):
    x, y, yaw = [start_pose[0]], [start_pose[1]], [start_pose[2]]
    for l, m in zip(lengths, mode):
        print(l, m)
        # 走过的总路程
        v = 0.0
        # 每次采样的路程
        step = 0.0
        if m == "S":
            # 如果是直线，每次走长度c弧长
            step = 0.5 * curvature
        else:
            # 如果不是直线，每次走gap弧长
            step = gap
        while l - v > step:
            # 沿着当前朝向直线运动step / c真实距离，如果是曲线运动，由于gap较小，可以近似为直线
            new_x = x[-1] + step / curvature * np.cos(yaw[-1])
            new_y = y[-1] + step / curvature * np.sin(yaw[-1])
            new_yaw = 0.0
            if m == "S":
                new_yaw = yaw[-1]
            elif m == "L":
                new_yaw = mod2Pi(yaw[-1] + step)
            else:
                new_yaw = mod2Pi(yaw[-1] - step)
            x.append(new_x)
            y.append(new_y)
            yaw.append(new_yaw)
            v += step
        # 计算剩下的路程
        re = l - v
        print(re)
        if re > 1e-6:
            new_x = x[-1] + re / curvature * np.cos(yaw[-1])
            new_y = y[-1] + re / curvature * np.sin(yaw[-1])
            new_yaw = 0.0
            if m == "S":
                new_yaw = yaw[-1]
            elif m == "L":
                new_yaw = mod2Pi(yaw[-1] + re)
            else:
                new_yaw = mod2Pi(yaw[-1] - re)
            x.append(new_x)
            y.append(new_y)
            yaw.append(new_yaw)
    return x, y, yaw

# 规划dubins path
def dubinsPlanning(start_pose, end_pose, curvature):
    # 第一步进行坐标转换，转换到以start到end连线为x轴，以start为原点的坐标系下
    coordinate = [start_pose[0], start_pose[1], np.arctan2(end_pose[1] - start_pose[1], end_pose[0] - start_pose[0])]
    transformed_start_pose = transformCoordinate(coordinate, start_pose)
    transformed_end_pose = transformCoordinate(coordinate, end_pose)
    # 带入原始dubins曲线规划
    x, y, yaw = rawPlanning(transformed_start_pose, transformed_end_pose, curvature)
    # 转换到原来的坐标系下
    raw_path = []
    for transformed in zip(x, y, yaw):
        print(transformed)
        raw_pose = retransfromCoordinate(coordinate, transformed)
        raw_path.append(raw_pose)
    #print(raw_path)
    return np.array(raw_path)
# 主函数
def main():
    # 初始化起点和终点
    start_pose = [1.0, 1.0, np.deg2rad(45.0)]
    end_pose = [-3.0, -3.0, np.deg2rad(-45.0)]
    # 初始化最小曲率
    curvature = 1.0
    # 开始规划
    path = dubinsPlanning(start_pose, end_pose, curvature)
    plt.plot(path.T[0], path.T[1])
    plt.show()

if __name__ == "__main__":
    main()