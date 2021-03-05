#! /usr/bin/python3
#! -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import map_loading
import grid_astar
import copy

# 测试函数1
def test1():
    # 设定起点和终点
    start = grid_astar.Point(2.5, 11.0)
    goal = grid_astar.Point(2.5, 21.0)
    # 得到初始设定
    config = grid_astar.Config()
    config.resolution_ = 0.5
    # 加载障碍物信息
    map_loader = map_loading.MapLoader()
    path = '/home/flztiii/pythonrobotic_myself/smoothed_astar/data/uturn_4.8_3.5_1.5.png'
    raw_obstacles = map_loader.loadMap(path)
    # 对障碍物进行过滤,在同一个栅格内的障碍物忽略
    obstacles = []
    for obstacle in raw_obstacles:
        new_obstacle = grid_astar.Point(int(obstacle.x_ / config.resolution_) * config.resolution_, int(obstacle.y_ / config.resolution_) * config.resolution_)
        if not grid_astar.Tools.isInPointSet(new_obstacle, obstacles):
            obstacles.append(new_obstacle)
    print(len(raw_obstacles), len(obstacles))
    # 对当前信息进行可视化
    obstalces_x, obstacles_y = grid_astar.Tools.departPoints(obstacles)
    plt.plot(obstalces_x, obstacles_y, ".k")
    plt.plot(start.x_, start.y_, "og")
    plt.plot(goal.x_, goal.y_, "xb")
    plt.grid(True)
    plt.axis("equal")
    # 构建A星规划器
    astar_planner = grid_astar.AstarPlanner(start, goal, obstacles, config)
    planned_path = astar_planner.planning()
    # 可视化规划路径
    path_x, path_y = grid_astar.Tools.departPoints(planned_path)
    # 对路径进行平滑
    smoother = grid_astar.Smoother()
    if len(planned_path) > 5:
        smoothed_path = smoother.smoothing(planned_path, obstacles)
        # 可视化平滑路径
        spath_x, spath_y = grid_astar.Tools.departPoints(smoothed_path)
        plt.cla()
        plt.plot(spath_x, spath_y, "-g")
    plt.plot(obstalces_x, obstacles_y, ".k")
    plt.plot(start.x_, start.y_, "og")
    plt.plot(goal.x_, goal.y_, "xb")
    plt.grid(True)
    plt.axis("equal")
    plt.plot(path_x, path_y, "-r")
    plt.show()

# 测试函数2
def test2():
    # 设定起点和终点
    start = grid_astar.Point(2.5, 7.5)
    goal = grid_astar.Point(2.5, 21.0)
    # 得到初始设定
    config = grid_astar.Config()
    config.resolution_ = 0.5
    # 加载障碍物信息
    map_loader = map_loading.MapLoader()
    path = '/home/flztiii/pythonrobotic_myself/smoothed_astar/data/uturn_6.5_4._1.5.png'
    raw_obstacles = map_loader.loadMap(path)
    # 对障碍物进行过滤,在同一个栅格内的障碍物忽略
    obstacles = []
    for obstacle in raw_obstacles:
        new_obstacle = grid_astar.Point(int(obstacle.x_ / config.resolution_) * config.resolution_, int(obstacle.y_ / config.resolution_) * config.resolution_)
        if not grid_astar.Tools.isInPointSet(new_obstacle, obstacles):
            obstacles.append(new_obstacle)
    print(len(raw_obstacles), len(obstacles))
    # 对当前信息进行可视化
    obstalces_x, obstacles_y = grid_astar.Tools.departPoints(obstacles)
    plt.plot(obstalces_x, obstacles_y, ".k")
    plt.plot(start.x_, start.y_, "og")
    plt.plot(goal.x_, goal.y_, "xb")
    plt.grid(True)
    plt.axis("equal")
    # 构建A星规划器
    astar_planner = grid_astar.AstarPlanner(start, goal, obstacles, config)
    planned_path = astar_planner.planning()
    # 可视化规划路径
    path_x, path_y = grid_astar.Tools.departPoints(planned_path)
    # 对路径进行平滑
    smoother = grid_astar.Smoother()
    show_path = []
    if len(planned_path) > 5:
        smoothed_path = smoother.smoothing(planned_path, obstacles)
        # 可视化平滑路径
        spath_x, spath_y = grid_astar.Tools.departPoints(smoothed_path)
        show_path = copy.deepcopy(smoothed_path)
        plt.cla()
        plt.plot(spath_x, spath_y, "-g")
    plt.plot(obstalces_x, obstacles_y, ".k")
    plt.plot(start.x_, start.y_, "og")
    plt.plot(goal.x_, goal.y_, "xb")
    plt.grid(True)
    plt.axis("equal")
    plt.plot(path_x, path_y, "-r")
    # 可视化路径的角度变化
    spath_x, spath_y = grid_astar.Tools.departPoints(show_path)
    yaws = []
    for i in range(0, len(spath_x) - 1):
        yaw = np.arctan2(spath_y[i + 1] - spath_y[i], spath_x[i + 1] - spath_x[i])
        yaws.append(yaw)
    yaws.append(yaws[-1])
    fig = plt.figure()
    plt.plot(np.linspace(0, len(yaws), len(yaws)), yaws)
    # 对路径的角度进行平滑
    sigma_s, simga_yaw = 5.0, 0.5
    blurred_yaws = smoother.yawBlur(show_path, sigma_s, simga_yaw)
    fig = plt.figure()
    plt.plot(np.linspace(0, len(blurred_yaws), len(blurred_yaws)), blurred_yaws)
    plt.show()

# 测试函数3
def test3():
    # 设定起点和终点
    start = grid_astar.Point(33, 5)
    goal = grid_astar.Point(133, 148)
    # 得到初始设定
    config = grid_astar.Config()
    config.resolution_ = 1.0
    # 加载障碍物信息
    map_loader = map_loading.MapLoader()
    path = '/home/flztiii/pythonrobotic_myself/smoothed_astar/data/scurve_with_cones.png'
    raw_obstacles = map_loader.loadMap(path)
    # 对障碍物进行过滤,在同一个栅格内的障碍物忽略
    obstacles = []
    for obstacle in raw_obstacles:
        new_obstacle = grid_astar.Point(int(obstacle.x_ / config.resolution_) * config.resolution_, int(obstacle.y_ / config.resolution_) * config.resolution_)
        if not grid_astar.Tools.isInPointSet(new_obstacle, obstacles):
            obstacles.append(new_obstacle)
    print(len(raw_obstacles), len(obstacles))
    # 对当前信息进行可视化
    obstalces_x, obstacles_y = grid_astar.Tools.departPoints(obstacles)
    plt.plot(obstalces_x, obstacles_y, ".k")
    plt.plot(start.x_, start.y_, "og")
    plt.plot(goal.x_, goal.y_, "xb")
    plt.grid(True)
    plt.axis("equal")
    # 构建A星规划器
    astar_planner = grid_astar.AstarPlanner(start, goal, obstacles, config)
    planned_path = astar_planner.planning()
    # 可视化规划路径
    path_x, path_y = grid_astar.Tools.departPoints(planned_path)
    # 对路径进行平滑
    show_path = []
    smoother = grid_astar.Smoother()
    if len(planned_path) > 5:
        smoothed_path = smoother.smoothing(planned_path, obstacles)
        # 可视化平滑路径
        spath_x, spath_y = grid_astar.Tools.departPoints(smoothed_path)
        show_path = copy.deepcopy(smoothed_path)
        plt.cla()
        plt.plot(spath_x, spath_y, "-g")
    plt.plot(obstalces_x, obstacles_y, ".k")
    plt.plot(start.x_, start.y_, "og")
    plt.plot(goal.x_, goal.y_, "xb")
    plt.grid(True)
    plt.axis("equal")
    plt.plot(path_x, path_y, "-r")
    # 可视化路径的角度变化
    spath_x, spath_y = grid_astar.Tools.departPoints(show_path)
    yaws = []
    for i in range(0, len(spath_x) - 1):
        yaw = np.arctan2(spath_y[i + 1] - spath_y[i], spath_x[i + 1] - spath_x[i])
        yaws.append(yaw)
    yaws.append(yaws[-1])
    fig = plt.figure()
    plt.plot(np.linspace(0, len(yaws), len(yaws)), yaws)
    # 对路径的角度进行平滑
    sigma_s, simga_yaw = 5.0, 0.4
    blurred_yaws = smoother.yawBlur(show_path, sigma_s, simga_yaw)
    fig = plt.figure()
    plt.plot(np.linspace(0, len(blurred_yaws), len(blurred_yaws)), blurred_yaws)
    plt.show()

if __name__ == "__main__":
    test1()