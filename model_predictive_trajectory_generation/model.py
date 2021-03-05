#！ /usr/bin/python3
# -*- coding: utf-8 -*-

"""

motion model of vehicle
author: flztiii

"""

import math
import scipy.interpolate
import numpy as np

# 全局变量
LEN = 1.0  # 车辆前后轴距
VEL = 10 / 3.6  # 车辆速度10km/h
ds = 0.1  # 车辆生成轨迹点采样间隔

# 车辆模型状态，用的是后轴中心的阿克曼模型
class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x_ = x  # 车辆位置x坐标
        self.y_ = y  # 车辆位置y坐标
        self.yaw_ = yaw  # 车辆后轴中心的朝向
        self.v_ = v  # 车辆的速度

# 弧度转化，将弧度转化到[-π,π]之间
def pi2Pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

# 更新车辆状态
def update(state, v, delta, dt, L):
    """
        :param state: 当前状态
        :param v: 速度
        :param delta: 前轮转角
        :param dt: 时间间隔
        :param L: 前后轮轴距
    """

    new_state = State()
    new_state.v_ = v
    new_state.x_ = state.x_ + v * dt * math.cos(state.yaw_)
    new_state.y_ = state.y_ + v * dt * math.sin(state.yaw_)
    new_state.yaw_ = state.yaw_ + v * math.tan(delta) / L * dt
    new_state.yaw_ = pi2Pi(new_state.yaw_)
    return new_state

# 生成路径
def generateTrajectory(s, k0, km, kf):
    """
        :param s: 生成轨迹的长度
        :param k0: 初始时刻前轮转角
        :param km: 中间时刻前轮转角
        :param kf: 结束时刻前轮转角
    """

    # 首先生成前轮转角随时间的变化关系
    # 根据泰勒公式，在较短时间内可以假设前轮转角随时间变化可以通过多项式表示，这里使用二次多项式
    time = s / VEL  # 得到轨迹总时长
    n = s / ds  # 得到轨迹的总采样点数
    dt = float(time / n)  # 每个采样点之间的时间间隔
    tk = np.array([0.0, time * 0.5, time])
    kk = np.array([k0, km, kf])
    t = np.arange(0.0, time, time / n)
    fkp = scipy.interpolate.interp1d(tk, kk, kind="quadratic")  # 得到二次表达式
    # 进行采样
    kp = [fkp(it) for it in t]
    # 根据得到的前轮转角随时间变化采样点得到车辆轨迹
    state = State()
    x, y, yaw = [state.x_], [state.y_], [state.yaw_]
    for i in range(0, len(kp)):
        state = update(state, VEL, kp[i], dt, LEN)
        x.append(state.x_)
        y.append(state.y_)
        yaw.append(state.yaw_)
    return x, y, yaw

# 生成轨迹的最后一个点状态
def generateFinalState(s, k0, km, kf):
    """
        :param s: 生成轨迹的长度
        :param k0: 初始时刻前轮转角
        :param km: 中间时刻前轮转角
        :param kf: 结束时刻前轮转角
    """
    
    # 首先生成前轮转角随时间的变化关系
    # 根据泰勒公式，在较短时间内可以假设前轮转角随时间变化可以通过多项式表示，这里使用二次多项式
    time = s / VEL  # 得到轨迹总时长
    n = s / ds  # 得到轨迹的总采样点数
    dt = float(time / n)  # 每个采样点之间的时间间隔
    tk = np.array([0.0, time * 0.5, time])
    kk = np.array([k0, km, kf])
    t = np.arange(0.0, time, time / n)
    fkp = scipy.interpolate.interp1d(tk, kk, "quadratic")  # 得到二次表达式
    # 进行采样
    kp = [fkp(it) for it in t]

    # 根据得到的前轮转角随时间变化采样点得到车辆轨迹
    state = State()
    for i in range(0, len(kp)):
        state = update(state, VEL, kp[i], dt, LEN)
    return state.x_, state.y_, state.yaw_