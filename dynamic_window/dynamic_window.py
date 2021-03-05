#! /usr/bin/python3
# -*- coding: utf-8 -*-

"""

dynamic window approach

author: flztiii

"""

import math
import numpy as np
import matplotlib.pyplot as plt
import copy

# 机器人状态信息
class RobotState:
    def __init__(self, x, y, yaw, v, w):
        self.x = x  # 横坐标
        self.y = y  # 纵坐标
        self.yaw = yaw  # 朝向角
        self.v = v  # 速度标量
        self.w = w  #角速度标量

# 机器人运动信息
class Motion:
    def __init__(self, v, w):
        self.v = v  # 速度标量
        self.w = w  #角速度标量

# 障碍物信息
class Obstacle:
    def __init__(self, x, y):
        self.x = x  # 横坐标
        self.y = y  # 纵坐标

# 目标点信息
class Goal:
    def __init__(self, x, y):
        self.x = x  # 横坐标
        self.y = y  # 纵坐标

# 机器人参数
class RobotConfig:
    def __init__(self):
        self.max_velocity = 1.0  # 最大速度
        self.min_velocity = -0.5  # 最小速度
        self.max_acceleration = 0.2  # 最大加速度
        self.max_yaw_rate = 40.0 * math.pi / 180.0  # 最大转向角
        self.max_yaw_rate_change = 40.0 * math.pi / 180.0  # 最大转向角变化率
        self.velocity_gap = 0.01  # 速度分辨率
        self.yaw_rate_gap = 0.1 * math.pi / 180.0 # 转向角分辨率
        self.time_gap = 0.1  # 时间窗口
        self.time_prediction = 3.0  # 预测时间
        self.robot_size = 1.0  # 机器人大小
        self.heading_cost_weight = 0.15  # 朝向权重
        self.velocity_cost_weight = 1.0  # 速度权重
        self.obstacle_cost_weight = 1.0  # 离障碍物距离权重  

# 动态窗口
class DynamicWindow:
    def __init__(self, min_velocity, max_velocity, min_yaw_rate, max_yaw_rate):
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.min_yaw_rate = min_yaw_rate
        self.max_yaw_rate = max_yaw_rate
    
    def calcInteraction(self, dyanmic_window):
        self.min_velocity = max(self.min_velocity, dyanmic_window.min_velocity)
        self.max_velocity = min(self.max_velocity, dyanmic_window.max_velocity)
        self.min_yaw_rate = max(self.min_yaw_rate, dyanmic_window.min_yaw_rate)
        self.max_yaw_rate = min(self.max_yaw_rate, dyanmic_window.max_yaw_rate)

# 获得滑动窗口
def getDynamicWindow(current_motion, config):
    Vs = DynamicWindow(config.min_velocity, config.max_velocity, -config.max_yaw_rate, config.max_yaw_rate)
    Vd = DynamicWindow(current_motion.v - config.time_gap * config.max_acceleration, 
                       current_motion.v + config.time_gap * config.max_acceleration,
                       current_motion.w - config.time_gap * config.max_yaw_rate_change,
                       current_motion.w + config.time_gap * config.max_yaw_rate_change)
    # 求交集
    Vd.calcInteraction(Vs)
    return Vd

# 根据行为更新状态
def action(state, motion, t):
    new_state = state
    new_state.v = motion.v
    new_state.w = motion.w
    new_state.yaw = state.yaw + motion.w * t
    new_state.x = state.x + motion.v * math.cos(new_state.yaw) * t
    new_state.y = state.y + motion.v * math.sin(new_state.yaw) * t
    return new_state

# 预测轨迹
def predictTrajectory(current_state, predict_motion, config):
    predict_trajectory = [current_state]
    tmp_state = copy.copy(current_state)
    for tmp_t in np.arange(0, config.time_prediction, config.time_gap):
        tmp_state = action(tmp_state, predict_motion, config.time_gap)
        predict_trajectory.append(copy.copy(tmp_state))
    return predict_trajectory

# 计算朝向损失
def calcHeadingCost(final_state, goal):
    error_angle = math.atan2(goal.y - final_state.y, goal.x - final_state.x)
    return abs(error_angle - final_state.yaw)

# 计算障碍物损失
def calcObstacleCost(predict_trajectory, obstacles, config):
    # 计算预测轨迹与障碍物的最近距离
    skip_n = 2
    min_distance = float("inf")
    for i in range(0, len(predict_trajectory), skip_n):
        for j in range(0, len(obstacles)):
            dx = predict_trajectory[i].x - obstacles[j].x
            dy = predict_trajectory[i].y - obstacles[j].y
            distance = math.sqrt(dx**2 + dy**2)
            if distance <= min_distance:
                min_distance = distance
    cost = 0.0
    if min_distance > config.robot_size:
        cost = 1.0 / min_distance
    else:
        cost = float("inf")
    return cost

# 得到最优运动
def getBestMotion(current_state, dynamic_window, obstacles, goal, config):
    min_cost = float("inf")
    best_prediction_trajectory = [current_state]
    best_motion = Motion(0.0, 0.0)
    for temp_v in np.arange(dynamic_window.min_velocity, dynamic_window.max_velocity, config.velocity_gap):
        for temp_w in np.arange(dynamic_window.min_yaw_rate, dynamic_window.max_yaw_rate, config.yaw_rate_gap):
            # 得到预测行为和轨迹
            predict_motion = Motion(temp_v, temp_w);
            prediction_trajectory = predictTrajectory(current_state, predict_motion, config)
            # 根据预测轨迹计算损失
            heading_cost = config.heading_cost_weight * calcHeadingCost(prediction_trajectory[-1], goal)
            velocity_cost = config.velocity_cost_weight * (config.max_velocity - prediction_trajectory[-1].v)
            obstacle_cost = config.obstacle_cost_weight * calcObstacleCost(prediction_trajectory, obstacles, config)
            total_cost = heading_cost + velocity_cost + obstacle_cost
            if total_cost <= min_cost:
                min_cost = total_cost
                best_motion = predict_motion
                best_prediction_trajectory = prediction_trajectory
    return best_motion, best_prediction_trajectory

# 可视化当前状态
def plotArrow(x, y, yaw, length=0.5, width=0.1):
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw), head_length=width, head_width=width)
    plt.plot(x, y)

# 主函数
def main():
    # 配置信息
    config = RobotConfig()
    # 当前状态
    current_state = RobotState(0.0, 0.0, math.pi / 8.0, 0.0, 0.0)
    # 目标点
    goal = Goal(10.0, 10.0);
    # 障碍物集合
    obstacles = [Obstacle(-1, -1),
                 Obstacle(0, 2),
                 Obstacle(4.0, 2.0),
                 Obstacle(5.0, 4.0),
                 Obstacle(5.0, 5.0),
                 Obstacle(5.0, 6.0),
                 Obstacle(5.0, 9.0),
                 Obstacle(8.0, 9.0),
                 Obstacle(7.0, 9.0),
                 Obstacle(12.0, 12.0)]
    # 当前运动
    current_motion = Motion(current_state.v, current_state.w)
    # 轨迹保存
    trajectory = [copy.copy(current_state)]
    # 初始化信息完成开始规划
    while True:
        # 根据当前状态得到动态窗口
        dynamic_window = getDynamicWindow(current_motion, config)
        # 从动态窗口中得到最优的运动
        best_motion, prediction_trajectory = getBestMotion(current_state, dynamic_window, obstacles, goal, config)
        # 更新当前状态、运动和轨迹
        current_state = action(current_state, best_motion, config.time_gap)
        current_motion = best_motion
        trajectory.append(copy.copy(current_state))
        # 可视化每一个状态
        vis_pred_trajectory = []
        for i in range(0, len(prediction_trajectory)):
            vis_pred_trajectory.append([prediction_trajectory[i].x, prediction_trajectory[i].y])
        vis_pred_trajectory = np.array(vis_pred_trajectory)
        vis_obstacles = []
        for i in range(0, len(obstacles)):
            vis_obstacles.append([obstacles[i].x, obstacles[i].y])
        vis_obstacles = np.array(vis_obstacles)
        plt.cla()
        plt.plot(vis_pred_trajectory[:, 0], vis_pred_trajectory[:, 1], "-g")
        plt.plot(current_state.x, current_state.y, "xr")
        plt.plot(goal.x, goal.y, "xb")
        plt.plot(vis_obstacles[:, 0], vis_obstacles[:, 1], "ok")
        plotArrow(current_state.x, current_state.y, current_state.yaw)
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)
        # 判断是否到达终点
        distance_to_goal = math.sqrt((current_state.x - goal.x) ** 2 + (current_state.y - goal.y) ** 2)
        if distance_to_goal <= config.robot_size:
            print("Goal Reached")
            break;
    # 可视化全部轨迹
    print("Done")
    vis_trajectory = []
    for i in range(0, len(trajectory)):
        vis_trajectory.append([trajectory[i].x, trajectory[i].y])
    vis_trajectory = np.array(vis_trajectory)
    plt.plot(vis_trajectory[:, 0], vis_trajectory[:, 1], "-r")
    plt.pause(0.0001)
    plt.show()

if __name__ == "__main__":
    main();