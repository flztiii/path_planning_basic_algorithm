#! /usr/bin/python3
# -*- coding: utf-8 -*-

"""

Eta^3 spline trajectory generation
author: flztiii

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.collections import LineCollection

sys.path.append(os.path.dirname(os.getcwd() + "/" + __file__))

from eta3spline_path_planning import EtaSpline, EtaSplineSet

# 报错类
class MaxVelocityNotReached(Exception):
    def __init__(self, actual_vel, max_vel):
        self.message = 'Actual velocity {} does not equal desired max velocity {}!'.format(
            actual_vel, max_vel)

# eta^3 spline trajectory类
class EtaSplineTrajectory(EtaSplineSet):
    def __init__(self, spline_segments, init_vel, init_acc, max_vel, max_acc, max_jerk):
        # 验证输入的正确性
        assert init_vel <= max_vel and init_vel >= 0
        assert init_acc >= 0 and init_acc <= max_acc
        # 父类初始化
        super(EtaSplineTrajectory, self).__init__(spline_segments=spline_segments)
        # 其他属性初始化
        self.v0_ = init_vel  # 初始速度
        self.a0_ = init_acc  # 初始加速度
        self.max_vel_ = max_vel  # 设定最大速度
        self.max_acc_ = max_acc  # 设定最大加速度
        self.max_jerk_ = max_jerk  # 设定加速度最大变化率
        # 生成信息
        self.segment_lengths_ = np.cumsum([self.spline_segments_[i].total_distance_ for i in range(0, self.n_)])  # 到达每一段曲线结束点的总路程长度
        # print(self.segment_lengths_)
        self.total_length_ = self.segment_lengths_[-1]
        print("total distance", self.total_length_)
        # 根据上述信息生成速度信息
        self.__velocityGeneration()
        # 记录上一次的index和u
        self.prev_index_ = 0
        self.prev_u_ = 0.0

    # 根据信息生成速度
    def __velocityGeneration(self):
        '''                   /~~~~~----------------~~~~~\
                             /                            \
                            /                              \
                           /                                \
                          /                                  \
        (v=v0, a=a0) ~~~~~                                    \
                                                               \
                                                                \~~~~~ (vf=0, af=0)
                     pos.|pos.|neg.|   cruise at    |neg.| neg. |neg.
                     max |max.|max.|     max.       |max.| max. |max.
                     jerk|acc.|jerk|    velocity    |jerk| acc. |jerk
            index     0    1    2      3 (optional)   4     5     6
        速度变化分为了七个阶段，要分别进行讨论

        '''

        # 首先计算加速度从初始值变化为最大值的过程
        t_s1 = (self.max_acc_ - self.a0_) / self.max_jerk_
        v_s1 = self.v0_ + 0.5 * self.max_jerk_ * t_s1 ** 2
        s_s1 = self.v0_ * t_s1 + 1.0/6.0 * self.max_jerk_ * t_s1 ** 3
        # 计算最后阶段加速度和速度同时降为0的过程
        t_sf = self.max_acc_ / self.max_jerk_
        v_sf = 0.5 * self.max_jerk_ * t_sf ** 2
        s_sf = 1.0/6.0 * self.max_jerk_ * t_sf ** 3
        
        # 根据计算出的上述信息判断能够达到的最大速度
        # solve for the maximum achievable velocity based on the kinematic limits imposed by max_acc_ and max_jerk_
        # this leads to a quadratic equation in v_max: a*v_max**2 + b*v_max + c = 0
        a = 1 / self.max_acc_
        b = 3. * self.max_acc_ / (2. * self.max_jerk_) + v_s1 / self.max_acc_ - (
            self.max_acc_**2 / self.max_jerk_ + v_s1) / self.max_acc_
        c = s_s1 + s_sf - self.total_length_ - 7. * self.max_acc_**3 / (3. * self.max_jerk_**2) \
            - v_s1 * (self.max_acc_ / self.max_jerk_ + v_s1 / self.max_acc_) \
            + (self.max_acc_**2 / self.max_jerk_ + v_s1 /
               self.max_acc_)**2 / (2. * self.max_acc_)
        v_max = (-b + np.sqrt(b**2 - 4. * a * c)) / (2. * a)
        
        # 如果计算出的v_max超过了设定的max_vel则说明可达到设定值，反之则无法达到设定值
        # 无法达到设定值的情况下，只能将设定值变为v_max
        if v_max < self.max_vel_:
            self.max_vel_ = v_max
        
        # 下一步，开始计算整个运动过程
        self.times_ = np.zeros(7)  # 七个阶段中每个阶段所花费的时间
        self.velocitys_ = np.zeros(7)  # 七个阶段中每个阶段结束时的速度
        self.travels_ = np.zeros(7)  # 七个阶段中每个阶段所走过的路程
        # 第一个阶段，加速度以最大变化率达到最大加速度
        self.times_[0] = t_s1
        self.velocitys_[0] = v_s1
        self.travels_[0] = s_s1
        # 第二个阶段，速度以最大加速度达到特定速度
        v_s2 = self.max_vel_ - self.max_acc_ ** 2 / (2. * self.max_jerk_)
        t_s2 = (v_s2 - self.velocitys_[0]) / self.max_acc_
        s_s2 = 0.5 * (self.velocitys_[0] + v_s2) * t_s2
        self.times_[1] = t_s2
        self.velocitys_[1] = v_s2
        self.travels_[1] = s_s2
        # 第三个阶段，加速度以最大变化率变为0，并与此同时，速度由特定速度达到最大速度
        t_s3 = self.max_acc_ / self.max_jerk_
        v_s3 = self.velocitys_[1] + self.max_acc_ * t_s3 - 0.5 * self.max_jerk_ * t_s3 ** 2
        s_s3 = self.velocitys_[1] * t_s3 + 0.5 * self.max_acc_ * t_s3 ** 2 - 1.0/6.0 * self.max_jerk_ * t_s3 ** 3
        self.times_[2] = t_s3
        self.velocitys_[2] = v_s3
        self.travels_[2] = s_s3
        # 确认当前速度已经达到最大速度
        if not np.isclose(self.velocitys_[2], self.max_vel_):
            # 没有能够到达最大速度错误
            raise MaxVelocityNotReached(self.velocitys_[2], self.max_vel_)
        # 第四个阶段，以最大速度行驶，最后计算
        # 第五个阶段，加速度以最大变化率达到最大减速度
        t_s5 = self.max_acc_ / self.max_jerk_
        v_s5 = self.velocitys_[2] - 0.5 * self.max_jerk_ * t_s5 ** 2
        s_s5 = self.velocitys_[2] * t_s5 - 1.0/6.0 * self.max_jerk_ * t_s5 ** 3
        self.times_[4] = t_s5
        self.velocitys_[4] = v_s5
        self.travels_[4] = s_s5
        # 第六个阶段，以最大减速度减速到特定速度
        v_s6 = self.max_acc_ ** 2 / (2. * self.max_jerk_)
        t_s6 = -(v_s6 - self.velocitys_[4]) / self.max_acc_
        s_s6 = 0.5 * (v_s6 + self.velocitys_[4]) * t_s6
        self.times_[5] = t_s6
        self.velocitys_[5] = v_s6
        self.travels_[5] = s_s6
        # 最后阶段，变加速运动加速度以最大变化率达到0
        t_s7 = self.max_acc_ / self.max_jerk_
        v_s7 = self.velocitys_[5] - self.max_acc_ * t_s7 + 0.5 * self.max_jerk_ * t_s7 ** 2
        s_s7 = self.velocitys_[5] * t_s7 - 0.5 * self.max_acc_ * t_s7 ** 2 + 1.0/6.0 * self.max_jerk_ * t_s7 ** 3
        self.times_[6] = t_s7
        self.velocitys_[6] = v_s7
        self.travels_[6] = s_s7
        # 验证最后速度是否为0
        if not np.isclose(self.velocitys_[6], 0.0):
            # 没有能够到达0速度错误
            raise MaxVelocityNotReached(self.velocitys_[6], 0.0)
        # 计算第四个阶段的持续时间
        assert self.total_length_ >= np.sum(self.travels_)
        s_s4 = self.total_length_ - np.sum(self.travels_)
        t_s4 = s_s4 / self.max_vel_
        v_s4 = self.max_vel_
        self.times_[3] = t_s4
        self.velocitys_[3] = v_s4
        self.travels_[3] = s_s4
        # 速度属性计算完成
        # 计算轨迹的总时长
        self.total_time_ = np.sum(self.times_)
        print("total time", self.total_time_)
        print("times", self.times_)
        print("velocitys", self.velocitys_)
        print("travelled distance", self.travels_)

    # 计算对应时间t的状态
    def calcState(self, t):
        # 首先计算时间t对应的路程和速度
        s, v, a = self.__timeToSVA(t)
        # 再计算对应曲线段index和参数u
        index, u = self.__sToIndexU(s)
        # print(u)
        # 计算t时刻对应的坐标
        x, y = self.spline_segments_[index].calcPosition(u)
        # 计算一阶导数
        dxu, dyu = self.spline_segments_[index].calcDerivative(u)
        # 计算二阶导数
        ddxu, ddyu = self.spline_segments_[index].calc2Derivative(u)
        # 计算对应的朝向
        yaw = np.arctan2(dyu, dxu)
        # 计算对应的角速度, 角速度的计算公式为[ddy(t)*dx(t) - ddx(t) * dy(t)] / v ** 2
        dsu = self.spline_segments_[index].funcSDerivative(u)
        yaw_rate = 0.0
        if not (v == 0 or dsu == 0):
            dut = v / dsu
            ddut = a / dsu - (dxu * ddxu + dyu * ddyu) * dut / dsu ** 3
            dyt = dyu * dut
            dxt = dxu * dut
            ddyt = ddyu * dut ** 2 + dyu * ddut
            ddxt = ddxu * dut ** 2 + dxu * ddut
            yaw_rate = (ddyt *dxt - ddxt * dyt) / v ** 2
        else:
            yaw_rate = 0.0
        state = np.array([x, y, yaw, v, yaw_rate])
        return state

    # 计算时间t对应的路程s和速度v
    def __timeToSVA(self, t):
        result = 0.0
        velocity = 0.0
        # 首先找到时间t对应的路程s
        if t < np.sum(self.times_[:1]):
            # 处于阶段一,加速度以最大变化率达到最大加速度
            result = self.v0_ * t + 1.0/6.0 * self.max_jerk_ * t ** 3
            velocity = self.v0_ + 0.5 * self.max_jerk_ * t ** 2
            acceleration = self.max_jerk_ * t
        elif t < np.sum(self.times_[:2]):
            # 处于阶段二,速度以最大加速度达到特定速度
            t = t - np.sum(self.times_[:1])
            result = np.sum(self.travels_[:1]) + self.velocitys_[0] * t + 0.5 * self.max_acc_ * t ** 2
            velocity = self.velocitys_[0] + self.max_acc_ * t
            acceleration = self.max_acc_
        elif t < np.sum(self.times_[:3]):
            # 处于阶段三,加速度以最大变化率变为0
            t = t - np.sum(self.times_[:2])
            result = np.sum(self.travels_[:2]) + self.velocitys_[1] * t + 0.5 * self.max_acc_ * t ** 2 - 1.0/6.0 * self.max_jerk_ * t ** 3
            velocity = self.velocitys_[1] + self.max_acc_ * t - 0.5 * self.max_jerk_ * t ** 2
            acceleration = self.max_acc_ - self.max_jerk_ * t
        elif t < np.sum(self.times_[:4]):
            # 处于阶段四,以最大速度行驶
            t = t - np.sum(self.times_[:3])
            result = np.sum(self.travels_[:3]) + self.velocitys_[2] * t
            velocity = self.velocitys_[2]
            acceleration = 0.0
        elif t < np.sum(self.times_[:5]):
            # 处于阶段五,加速度以最大变化率达到最大减速度
            t = t - np.sum(self.times_[:4])
            result = np.sum(self.travels_[:4]) + self.velocitys_[2] * t - 1.0/6.0 * self.max_jerk_ * t ** 3
            velocity = self.velocitys_[3] - 0.5 * self.max_jerk_ * t ** 2
            acceleration = - self.max_jerk_ * t
        elif t < np.sum(self.times_[:6]):
            # 处于阶段六,以最大减速度减速到特定速度
            t = t - np.sum(self.times_[:5])
            result = np.sum(self.travels_[:5]) + self.velocitys_[4] * t - 0.5 * self.max_acc_ * t ** 2
            velocity = self.velocitys_[4] - self.max_acc_ * t
            acceleration = - self.max_acc_
        elif t < np.sum(self.times_[:7]):
            # 处于阶段七,变加速运动加速度以最大变化率达到0
            t = t - np.sum(self.times_[:6])
            result = np.sum(self.travels_[:6]) + self.velocitys_[5] * t - 0.5 * self.max_acc_ * t ** 2 + 1.0/6.0 * self.max_jerk_ * t ** 3
            velocity = self.velocitys_[5] - self.max_acc_ * t + 0.5 * self.max_jerk_ * t ** 2
            acceleration = - self.max_acc_ + self.max_jerk_ * t
        else:
            result = self.total_length_
            velocity = 0.0
            acceleration = 0.0
        return result, velocity, acceleration

    # 根据路程s计算对应的曲线段下标index和对应u
    def __sToIndexU(self, s):
        assert s <= self.total_length_
        # 首先判断下标index
        index = 0
        for i in range(0, self.n_):
            if i < self.n_ - 1:
                if self.segment_lengths_[i] > s:
                    index = i
                    break
            else:
                index = i
        # 得到index后再计算对应的参数u
        # 计算出在对应曲线段上的路程s
        if index > 0:
            s = s - self.segment_lengths_[index - 1]
        # 根据s(u)的方程与导数，利用牛顿迭代方法得到u
        # 判断是否可以使用之前的u作为迭代初始值
        init_u = 0.0
        if index == self.prev_index_:
            init_u = self.prev_u_
        u = init_u
        while (u >= 0 and u <= 1) and np.abs(s - self.spline_segments_[index].funcS(u)) > 1e-3:
            u += (s - self.spline_segments_[index].funcS(u)) / self.spline_segments_[index].funcSDerivative(u)
        u = max(0, min(u, 1))
        # 更新上一次记录
        self.prev_index_ = index
        self.prev_u_ = u
        return index, u



# 主函数1，改变eta前两个参数对轨迹的影响
def main1():
    # 对于相同的边界条件使用不同的eta进行曲线的生成
    for i in range(0, 10):
        # 初始化eta
        eta = [i, i, 0, 0, 0, 0]  # eta为6维度自由变量
        # 初始化边界条件
        start_pos = [0, 0, 0, 0, 0]  # 起点位姿，形式为x坐标，y坐标，朝向yaw，曲率kappa，曲率导数
        end_pos = [4.0, 3.0, 0, 0, 0]  # 终点位姿，形式同上
        # 初始运动状态信息
        motion = [0.0, 0.0]
        # 车辆运动约束
        max_velocity = 0.5
        max_acceleration = 0.5
        max_jerk = 5.0
        # 构造eta spline
        eta_spline = EtaSpline(start_pos, end_pos, eta)
        # 构造eta spline trajectory
        eta_spline_segments = []
        eta_spline_segments.append(eta_spline)
        eta_spline_trajectory = EtaSplineTrajectory(eta_spline_segments, motion[0], motion[1], max_velocity, max_acceleration, max_jerk)
        # 对轨迹进行采样, 100个采样点，每个采样点包含5维度信息x,y,theta,v,w
        sample_states = np.zeros((5, 101))
        sample_times = np.linspace(0.0, eta_spline_trajectory.total_time_, sample_states.shape[1])
        for i, t in enumerate(sample_times):
            sample_states[:, i] = eta_spline_trajectory.calcState(t)
        # 进行可视化
        plt.plot(sample_states[0, :], sample_states[1, :])
        plt.axis("equal")
        plt.grid(True)
        plt.pause(1.0)
    plt.show()

# 主函数1，改变eta中间两个参数对轨迹的影响
def main1():
    # 对于相同的边界条件使用不同的eta进行曲线的生成
    for i in range(0, 10):
        # 初始化eta
        eta = [0.1, 0.1, (i - 5) * 20, (5 - i) * 20, 0, 0]  # eta为6维度自由变量
        # 初始化边界条件
        start_pos = [0, 0, 0, 0, 0]  # 起点位姿，形式为x坐标，y坐标，朝向yaw，曲率kappa，曲率导数
        end_pos = [4.0, 3.0, 0, 0, 0]  # 终点位姿，形式同上
        # 初始运动状态信息
        motion = [0.0, 0.0]
        # 车辆运动约束
        max_velocity = 0.5
        max_acceleration = 0.5
        max_jerk = 5.0
        # 构造eta spline
        eta_spline = EtaSpline(start_pos, end_pos, eta)
        # 构造eta spline trajectory
        eta_spline_segments = []
        eta_spline_segments.append(eta_spline)
        eta_spline_trajectory = EtaSplineTrajectory(eta_spline_segments, motion[0], motion[1], max_velocity, max_acceleration, max_jerk)
        # 对轨迹进行采样, 100个采样点，每个采样点包含5维度信息x,y,theta,v,w
        sample_states = np.zeros((5, 101))
        sample_times = np.linspace(0.0, eta_spline_trajectory.total_time_, sample_states.shape[1])
        for i, t in enumerate(sample_times):
            sample_states[:, i] = eta_spline_trajectory.calcState(t)
        # 进行可视化
        plt.plot(sample_states[0, :], sample_states[1, :])
        plt.axis("equal")
        plt.grid(True)
        plt.pause(1.0)
    plt.show()

# 主函数3，论文中曲线给出的参数
def main3():
    eta_spline_segments = []
    # 初始运动状态信息
    motion = [0.0, 0.0]
    # 车辆运动约束
    max_velocity = 2.0
    max_acceleration = 0.5
    max_jerk = 1.0
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

    # 构造eta spline trajectory
    eta_spline_trajectory = EtaSplineTrajectory(eta_spline_segments, motion[0], motion[1], max_velocity, max_acceleration, max_jerk)
    # 对轨迹进行采样, 100个采样点，每个采样点包含5维度信息x,y,theta,v,w
    sample_states = np.zeros((5, 100))
    sample_times = np.linspace(0.0, eta_spline_trajectory.total_time_, sample_states.shape[1])
    for i, t in enumerate(sample_times):
        sample_states[:, i] = eta_spline_trajectory.calcState(t)
    # 采样完成进行可视化
    fig, ax = plt.subplots()
    x, y = sample_states[0, :], sample_states[1, :]
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segs, cmap=plt.get_cmap('inferno'))
    ax.set_xlim(np.min(x) - 1, np.max(x) + 1)
    ax.set_ylim(np.min(y) - 1, np.max(y) + 1)
    lc.set_array(sample_states[3, :])
    lc.set_linewidth(3)
    ax.add_collection(lc)
    axcb = fig.colorbar(lc)
    axcb.set_label('velocity(m/s)')
    ax.set_title('Trajectory')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.pause(1.0)

    fig1, ax1 = plt.subplots()
    ax1.plot(sample_times, sample_states[3, :], 'b-')
    ax1.set_xlabel('time(s)')
    ax1.set_ylabel('velocity(m/s)', color='b')
    ax1.tick_params('y', colors='b')
    ax1.set_title('Control')
    ax2 = ax1.twinx()
    ax2.plot(sample_times, sample_states[4, :], 'r-')
    ax2.set_ylabel('angular velocity(rad/s)', color='r')
    ax2.tick_params('y', colors='r')
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main3()