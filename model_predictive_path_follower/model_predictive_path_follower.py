#! /usr/bin/python
#! -*- coding: utf-8 -*-

"""

model predictive path following 
author: flztiii

"""

import sys
sys.path.append("./cubic_spline/")
try:
    import cubic_spline
except:
    raise
import numpy as np
import matplotlib.pyplot as plt
import cvxpy
import copy

# 全局变量
DL = 1.0  # [m] 轨迹点采样间距
DT = 0.2  # [s] time tick
T = 6  # horizon length
NX = 4  # x = x, y, v, yaw
NU = 2  # a = [accel, steer]

GOAL_DIS = 1.5  # goal distance
STOP_SPEED = 0.5 / 3.6  # stop speed
MAX_TIME = 200.0  # max simulation time

# iterative paramter
MAX_ITER = 3  # Max iteration
DU_TH = 0.1  # iteration finish param

TARGET_SPEED = 10.0 / 3.6  # [m/s] target speed
N_IND_SEARCH = 10  # Search index number

# Vehicle parameters
LENGTH = 4.5  # [m]
WIDTH = 2.0  # [m]
BACKTOWHEEL = 1.0  # [m]
WHEEL_LEN = 0.3  # [m]
WHEEL_WIDTH = 0.2  # [m]
TREAD = 0.7  # [m]
WB = 2.5  # [m]

MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(30.0)  # maximum steering speed [rad/s]
MAX_SPEED = 55.0 / 3.6  # maximum speed [m/s]
MIN_SPEED = -20.0 / 3.6  # minimum speed [m/s]
MAX_ACCEL = 1.0  # maximum accel [m/ss]

def get_nparray_from_matrix(x):
    return np.array(x).flatten()

def pi_2_pi(angle):
    while(angle > np.pi):
        angle = angle - 2.0 * np.pi

    while(angle < -np.pi):
        angle = angle + 2.0 * np.pi

    return angle

def smoothYaw(yaw):
    for i in range(len(yaw) - 1):
        dyaw = yaw[i + 1] - yaw[i]

        while dyaw >= np.pi / 2.0:
            yaw[i + 1] -= np.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

        while dyaw <= -np.pi / 2.0:
            yaw[i + 1] += np.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

    return yaw    

# 判断是否到达终点
def checkGoal(state, goal_x, goal_y):
    goal_dis = np.sqrt((state[0] - goal_x)**2 + (state[1] - goal_y)**2)
    if goal_dis < GOAL_DIS and np.abs(state[3]) < STOP_SPEED:
        return True
    else:
        return False

def plotCar(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover

    outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[np.cos(yaw), np.sin(yaw)],
                     [-np.sin(yaw), np.cos(yaw)]])
    Rot2 = np.array([[np.cos(steer), np.sin(steer)],
                     [-np.sin(steer), np.cos(steer)]])

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(x, y, "*")

# 计算路径与当前状态最近点下标
def calcNearestIndex(path_x, path_y, state, last_index = 0):
    assert(len(path_x) == len(path_y) and len(path_x) > last_index)
    min_distance = float('inf')
    min_index = last_index
    for i in range(last_index, len(path_x)):
        distance = np.sqrt((state[0] - path_x[i])**2 + (state[1] - path_y[i])**2)
        if distance < min_distance:
            min_distance = distance
            min_index = i
    return min_index

# 获取参考路径
def getReference(path_x, path_y, path_yaw, path_speeds, state, start_index, steps):
    assert(start_index < len(path_x))
    ref_x, ref_y, ref_yaw, ref_speeds, ref_steers = list(), list(), list(), list(), list()
    for i in range(0, steps):
        index = start_index + int(round(state[-1] * DT * float(i) / DL))
        index = min(index, len(path_x) - 1)
        ref_x.append(path_x[index])
        ref_y.append(path_y[index])
        ref_yaw.append(path_yaw[index])
        ref_speeds.append(path_speeds[index])
        ref_steers.append(0.0)
    return ref_x, ref_y, ref_yaw, ref_speeds, ref_steers

# 更新状态
def updateState(state, action_a, action_t):
    # 初始化新状态
    new_state = [0.0] * len(state)
    # 防止输入越界
    if action_t >= MAX_STEER:
        action_t = MAX_STEER
    elif action_t <= -MAX_STEER:
        action_t = -MAX_STEER
    # 更新状态
    new_state[0] = state[0] + state[3] * np.cos(state[2]) * DT
    new_state[1] = state[1] + state[3] * np.sin(state[2]) * DT
    new_state[2] = state[2] + state[3]/ WB * np.tan(action_t) * DT
    new_state[3] = state[3] + action_a * DT
    # 防止状态越界
    if new_state[3] > MAX_SPEED:
        new_state[3] = MAX_SPEED
    elif state[3] < MIN_SPEED:
        new_state[3] = MIN_SPEED
    return new_state

# 进行模型状态预测
def modelPredictStates(state, actions_a, actions_t):
    pred_state = copy.deepcopy(state)
    pre_x = [pred_state[0]]
    pre_y = [pred_state[1]]
    pre_yaw = [pred_state[2]]
    pre_speeds = [pred_state[3]]
    for action_a, action_t in zip(actions_a, actions_t):
        pred_state = updateState(pred_state, action_a, action_t)
        pre_x.append(pred_state[0])
        pre_y.append(pred_state[1])
        pre_yaw.append(pred_state[2])
        pre_speeds.append(pred_state[3])
    return pre_x, pre_y, pre_yaw, pre_speeds

# 计算状态转移矩阵
def calcTransMatrix(yaw, speed, steer):
    A = np.zeros((NX, NX))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    A[3, 3] = 1.0
    A[0, 2] = DT * np.cos(yaw)
    A[0, 3] = - DT * speed * np.sin(yaw)
    A[1, 2] = DT * np.sin(yaw)
    A[1, 3] = DT * speed * np.cos(yaw)
    A[3, 2] = DT * np.tan(steer) / WB

    B = np.zeros((NX, NU))
    B[2, 0] = DT
    B[3, 1] = DT * speed / (np.cos(steer) ** 2 * WB)

    C = np.zeros(NX)
    C[0] = DT * speed * np.sin(yaw) * yaw
    C[1] = - DT * speed * np.cos(yaw) * yaw
    C[3] = - DT * speed * steer / (WB * np.cos(steer) ** 2)

    return A, B, C

# 进行模型预测优化
def modelPredictiveOptimization(ref_x, ref_y, ref_yaw, ref_speeds, ref_steers, pre_yaw, pre_speeds, state):
    # 使用cvxpy进行优化
    # 定义优化变量
    x = cvxpy.Variable((NX, T))
    u = cvxpy.Variable((NU, T - 1))
    # 定义损失函数
    cost = 0.0
    # 与参考路径的区别最小化
    for t in range(1, T):
        cost += (x[0, t] - ref_x[t])**2
        cost += (x[1, t] - ref_y[t])**2
        cost += 0.5 * (x[2, t] - ref_speeds[t])**2
        cost += 0.5 * (x[3, t] - ref_yaw[t])**2
    # 运动最小化
    for t in range(0, T - 1):
        cost += 0.01 * u[0, t]**2
        cost += 0.01 * u[1, t]**2
    # 运动变化最小化
    for t in range(0, T - 2):
        cost += 0.01 * (u[0, t+1] - u[0, t]) ** 2
        cost += (u[1, t+1] - u[1, t]) ** 2
    # 定义限制条件
    constraints = []
    # 初始状态限制
    constraints += [x[0, 0] == state[0]]
    constraints += [x[1, 0] == state[1]]
    constraints += [x[2, 0] == state[3]]
    constraints += [x[3, 0] == state[2]]
    # 最大速度限制
    constraints += [x[2, :] <= MAX_SPEED]
    # 最小速度限制
    constraints += [x[2, :] >= MIN_SPEED]
    # 最大加速度限制
    constraints += [cvxpy.abs(u[0, :]) <= MAX_ACCEL]
    # 转向限制
    constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]
    # 转向变化限制
    for t in range(0, T - 2):
        constraints += [cvxpy.abs(u[1, t+1] - u[1, t]) <= MAX_DSTEER * DT]
    # 状态转移限制
    for t in range(0, T - 1):
        # 计算状态转移矩阵
        A, B, C = calcTransMatrix(pre_yaw[t], pre_speeds[t], ref_steers[t])
        constraints += [x[:, t+1] == A * x[:, t] + B * u[:, t] + C]

    # 定义优化问题
    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    # 求解优化问题
    prob.solve(solver=cvxpy.ECOS, verbose=False)
    # 判断结果
    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        ox = get_nparray_from_matrix(x.value[0, :])
        oy = get_nparray_from_matrix(x.value[1, :])
        oyaw = get_nparray_from_matrix(x.value[3, :])
        ov = get_nparray_from_matrix(x.value[2, :])
        oa = get_nparray_from_matrix(u.value[0, :])
        ot = get_nparray_from_matrix(u.value[1, :])
    else:
        print("no solution for optimization")
        ox, oy, oyaw, ov, oa, ot = None, None, None, None, None, None
    return ox, oy, oyaw, ov, oa, ot


# 迭代进行模型预测控制
def iterModelPredictiveControls(ref_x, ref_y, ref_yaw, ref_speeds, ref_steers, state):
    assert(len(ref_x) == len(ref_y) and len(ref_y) == len(ref_yaw) and len(ref_yaw) == len(ref_speeds) and len(ref_speeds) == len(ref_steers))
    # 初始化控制
    actions_a = [0.0] * (len(ref_x) - 1) 
    actions_t = [0.0] * (len(ref_x) - 1)
    opt_x, opt_y, opt_yaw, opt_speeds = None, None, None, None
    for i in range(0, MAX_ITER):
        # 根据当前控制生成预测状态
        pre_x, pre_y, pre_yaw, pre_speeds = modelPredictStates(state, actions_a, actions_t)
        assert(len(pre_x) == len(ref_x))
        # 进行运动优化
        opt_x, opt_y, opt_yaw, opt_speeds, new_actions_a, new_actions_t = modelPredictiveOptimization(ref_x, ref_y, ref_yaw, ref_speeds, ref_steers, pre_yaw, pre_speeds, state)
        # 判断运动是否有效
        if new_actions_a is None:
            exit(0)
        # 判断运动的变化
        action_change = np.sum(np.array(actions_a) - np.array(new_actions_a)) + np.sum(np.array(actions_t) - np.array(new_actions_t))
        # 更新运动
        actions_a = new_actions_a
        actions_t = new_actions_t
        # 判断变化是否小于阈值
        if action_change < DU_TH:
            print("mpc update success")
            break
    else:
        print("up to max iteration")
    
    return opt_x, opt_y, opt_yaw, opt_speeds, actions_a, actions_t

# 路径跟随
def pathFollowing(fpath_x, fpath_y, fpath_yaw, fpath_speeds, init_state):
     # initial yaw compensation
    if init_state[2] - fpath_yaw[0] >= np.pi:
        init_state[2] -= np.pi * 2.0
    elif init_state[2] - fpath_yaw[0] <= -np.pi:
        init_state[2] += np.pi * 2.0
    # 定义当前状态
    current_state = init_state
    # 定义当前状态对应下标
    current_index = 0
    # 定义时间
    time = 0.0
    # 定义记录量
    x = [current_state[0]]
    y = [current_state[1]]
    yaw = [current_state[2]]
    v = [current_state[3]]
    t = [0.0]
    d = [0.0]
    a = [0.0]
    # 开始进行路径跟随
    while time < MAX_TIME:
        # 计算路径与当前状态最近点下标
        current_index = calcNearestIndex(fpath_x, fpath_y, current_state, current_index)
        # 计算参考路径
        ref_x, ref_y, ref_yaw, ref_speeds, ref_steers = getReference(fpath_x, fpath_y, fpath_yaw, fpath_speeds, current_state, current_index, T)
        # 迭代进行模型预测控制
        opt_x, opt_y, opt_yaw, opt_speeds, actions_a, actions_t = iterModelPredictiveControls(ref_x, ref_y, ref_yaw, ref_speeds, ref_steers, current_state)
        # 进行状态更新
        current_state = updateState(current_state, actions_a[0], actions_t[0])
        # 进行记录
        x.append(current_state[0])
        y.append(current_state[1])
        yaw.append(current_state[2])
        v.append(current_state[3])
        t.append(time)
        d.append(actions_a[0])
        a.append(actions_t[0])
        # 进行可视化
        plt.cla()
        if opt_x is not None:
            plt.plot(opt_x, opt_y, "xr", label="MPC")
        plt.plot(fpath_x, fpath_y, "-r", label="course")
        plt.plot(x, y, "ob", label="trajectory")
        plt.plot(ref_x, ref_y, "xk", label="xref")
        plt.plot(fpath_x[current_index], fpath_y[current_index], "xg", label="target")
        plotCar(current_state[0], current_state[1], current_state[2], steer=actions_t[0])
        plt.axis("equal")
        plt.grid(True)
        plt.title("Time[s]:" + str(round(time, 2))
                    + ", speed[km/h]:" + str(round(current_state[3] * 3.6, 2)))
        plt.pause(0.0001)
        # 判断是否到达终点
        if checkGoal(current_state, fpath_x[-1], fpath_y[-1]):
            print("Goal Reached")
            break
        # 增加时间
        time += DT
    return x, y, yaw, v, t, d, a

# 主函数
def main():
    # 给出路点
    # waypoints_x = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    # waypoints_y = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    waypoints_x = [0.0, 60.0, 125.0, 50.0, 75.0, 30.0, -10.0]
    waypoints_y = [0.0, 0.0, 50.0, 65.0, 30.0, 50.0, -20.0]
    # waypoints_x = [0.0, -10.0, -20.0, -40.0, -50.0, -60.0, -70.0]
    # waypoints_y = [0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0]
    # 生成引导轨迹
    spline = cubic_spline.CubicSpline2D(waypoints_x, waypoints_y)
    sample_s = np.arange(0.0, spline.s_[-1], DL)
    point_x, point_y = spline.calcPosition(sample_s)
    point_yaw = spline.calcYaw(sample_s)
    point_yaw = smoothYaw(point_yaw)
    point_kappa = spline.calcKappa(sample_s)
    # 生成对应速度
    speeds = [TARGET_SPEED] * len(point_x)
    speeds[-1] = 0.0
    # 定义初始状态
    init_state = [point_x[0], point_y[0], np.pi - point_yaw[0], 0.0]
    # 进行路径跟随
    x, y, yaw, v, t, d, a = pathFollowing(point_x, point_y, point_yaw, speeds, init_state)

    # 进行可视化
    plt.close("all")
    plt.subplots()
    plt.plot(point_x, point_y, "-r", label="spline")
    plt.plot(x, y, "-g", label="tracking")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.legend()

    plt.subplots()
    plt.plot(t, yaw, "-r", label="speed")
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("Yaw [rad]")

    plt.show()

if __name__ == "__main__":
    main()
