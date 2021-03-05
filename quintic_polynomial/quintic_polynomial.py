#! /usr/bin/python3
# -*- coding: utf-8 -*-

"""

Quintic Polynomial Planning
author: flztiii

"""

import numpy as np
import matplotlib.pyplot as plt

# 全局变量
MIN_T = 5.0  # 到达终点的最小时间开销
MAX_T = 100.0  # 到达终点的最大时间开销

# 五次多项式
class QuinticPolynomial:
    def __init__(self, params, T):
        self.params_ = params
        self.T_ = T
        # 生成多项式系数计算
        self.__coefficientGeneration()
        
    # 生成多项式系数计算
    def __coefficientGeneration(self):
        a0 = self.params_[0]
        a1 = self.params_[1]
        a2 = self.params_[2] * 0.5
        A = np.array([[self.T_ ** 3, self.T_ ** 4, self.T_ ** 5], [3. * self.T_ ** 2, 4. * self.T_ ** 3, 5. * self.T_ ** 4], [6. * self.T_, 12. * self.T_ ** 2, 20. * self.T_ ** 3]])
        b = np.array([self.params_[3] - a0 - a1 * self.T_ - a2 * self.T_ ** 2, self.params_[4] - a1 - 2. * a2 * self.T_, self.params_[5] - 2. * a2]).T
        x = np.linalg.solve(A, b)
        a3 = x[0]
        a4 = x[1]
        a5 = x[2]
        self.coefficient_ = np.array([a0, a1, a2, a3, a4, a5])

    # 给定输入时间，计算对应值
    def calcValue(self, t):
        return np.dot(self.coefficient_, np.array([1., t, t ** 2, t ** 3, t ** 4, t ** 5]).T)
    
    # 给定输入时间，计算导数
    def calcDerivation(self, t):
        return np.dot(self.coefficient_[1:], np.array([1., 2. * t, 3. * t ** 2, 4. * t ** 3, 5. * t ** 4]))
    
    # 给定输入时间，计算二阶导数
    def calc2Derivation(self, t):
        return np.dot(self.coefficient_[2:], np.array([2. , 6. * t, 12. * t ** 2, 20. * t ** 3]))
    
    # 给定输入时间，计算三阶导数
    def calc3Derivation(self, t):
        return np.dot(self.coefficient_[3:], np.array([6., 24. * t, 60. * t ** 2]))

# 五次多项式规划
def quinticPolynumialPlanning(start_pose, start_motion, goal_pose, goal_motion, max_accel, max_jerk, dt):
    # 初始化参数
    params_x = [start_pose[0], start_motion[0] * np.cos(start_pose[2]), start_motion[1] * np.cos(start_pose[2]), goal_pose[0], goal_motion[0] * np.cos(goal_pose[2]), goal_motion[1] * np.cos(goal_pose[2])]
    params_y = [start_pose[1], start_motion[0] * np.sin(start_pose[2]), start_motion[1] * np.sin(start_pose[2]), goal_pose[1], goal_motion[0] * np.sin(goal_pose[2]), goal_motion[1] * np.sin(goal_pose[2])]
    # 最终的路径，速度，加速度，加速度变化率在时间轴上的采样
    final_path, final_vels, final_accs, final_jerks, final_times = [], [], [], [], []
    # 遍历时间
    for T in np.arange(MIN_T, MAX_T, MIN_T):
        # 分别生成x，y的五次多项式
        xqp = QuinticPolynomial(params_x, T)
        yqp = QuinticPolynomial(params_y, T)
        print(T, xqp.coefficient_, yqp.coefficient_)
        # 对多项式进行采样
        path, vels, accs, jerks, times = [], [], [], [], []
        for sample_t in np.arange(0, T + dt, dt):
            # 计算对应时间的位置等信息
            sample_x = xqp.calcValue(sample_t)
            sample_dx = xqp.calcDerivation(sample_t)
            sample_ddx = xqp.calc2Derivation(sample_t)
            sample_dddx = xqp.calc3Derivation(sample_t)
            sample_y = yqp.calcValue(sample_t)
            sample_dy = yqp.calcDerivation(sample_t)
            sample_ddy = yqp.calc2Derivation(sample_t)
            sample_dddy = yqp.calc3Derivation(sample_t)
            # 采样点位置
            sample_pose = [sample_x, sample_y, np.arctan2(sample_dy, sample_dx)]
            path.append(sample_pose)
            # 采样点运动信息
            sample_vel = np.hypot(sample_dx, sample_dy)
            vels.append(sample_vel)
            sample_acc = np.hypot(sample_ddx, sample_ddy)  # 这里是加速度的大小，并不是速度大小的变化率，包括速度大小变化率和角速度变化率两部分
            if len(vels) > 1 and vels[-1] - vels[-2] < 0:
                sample_acc *= -1.0
            accs.append(sample_acc)
            sample_jerk = np.hypot(sample_dddx, sample_dddy)
            if len(accs) > 1 and accs[-1] - accs[-2] < 0:
                sample_jerk *= -1.0
            jerks.append(sample_jerk)
            times.append(sample_t)
        # 根据采样信息判断是否能够满足限制条件
        if max([np.abs(acc) for acc in accs]) <= max_accel and max([np.abs(jerk) for jerk in jerks]) <= max_jerk:
            # 满足限制条件
            final_path = np.array(path)
            final_vels = np.array(vels)
            final_accs = np.array(accs)
            final_jerks = np.array(jerks)
            final_times = np.array(times)
            print("find path")
            break
    return final_path, final_vels, final_accs, final_jerks, final_times

# 绘制箭头
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
    # 初始化边界条件和限制条件
    start_pose = [10.0, 10.0, np.deg2rad(10.0)]  # 起点位置x,y,yaw
    start_motion = [1.0, 0.1]  # 起点运动v,a
    goal_pose = [30.0, -10.0, np.deg2rad(20.0)]  # 终点位置
    goal_motion = [1.0, 0.1]  # 终点运动
    max_accel = 1.0  # max accel [m/ss]
    max_jerk = 0.5  # max jerk [m/sss]
    dt = 0.1  # time tick [s]
    # 开始进行规划
    final_path, final_vels, final_accs, final_jerks, final_times = quinticPolynumialPlanning(start_pose, start_motion, goal_pose, goal_motion, max_accel, max_jerk, dt)

    # 进行可视化
    # 首先可视化代理的运动过程
    for i, time in enumerate(final_times):
        plt.cla()  # 清楚之前的绘制信息
        plt.grid(True)  # 绘制网格
        plt.axis("equal")  # 绘制坐标
        plotArrow(*start_pose)  # 绘制起点
        plotArrow(*goal_pose)  # 绘制终点
        plotArrow(*(final_path[i]))  # 绘制当前位置
        plt.title("Time[s]:" + str(time)[0:4] +
                  " v[m/s]:" + str(final_vels[i])[0:4] +
                  " a[m/ss]:" + str(final_accs[i])[0:4] +
                  " jerk[m/sss]:" + str(final_jerks[i])[0:4],
                 )
        plt.pause(0.01)
    # 可视化曲线
    plt.plot(final_path[:, 0], final_path[:, 1])
    # 可视化朝向随时间变化曲线
    plt.subplots()
    plt.plot(final_times, [np.rad2deg(i) for i in final_path[:, 2]], "-r")
    plt.xlabel("Time[s]")
    plt.ylabel("Yaw[deg]")
    plt.grid(True)
    # 可视化速度随时间变化曲线
    plt.subplots()
    plt.plot(final_times, final_vels)
    plt.xlabel("Time[s]")
    plt.ylabel("Velocity[m/s]")
    plt.grid(True)
    # 可视化加速度随时间变化曲线
    plt.subplots()
    plt.plot(final_times, final_accs)
    plt.xlabel("Time[s]")
    plt.ylabel("Acceleration[m/s^2]")
    plt.grid(True)
    # 可视化加速度变化率随时间变化曲线
    plt.subplots()
    plt.plot(final_times, final_jerks)
    plt.xlabel("Time[s]")
    plt.ylabel("Jerk[m/s^3]")
    plt.grid(True)
    
    plt.show()


if __name__ == "__main__":
    main()