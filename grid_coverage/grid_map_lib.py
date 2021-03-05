#! /sur/bin/python3
# -*- coding: utf-8 -*-

"""

Grid map library in python

author: flztiii

"""

import matplotlib.pyplot as plt
import numpy as np

# 栅格地图
class GridMap:
    def __init__(self, center_px, center_py, width_nx, width_ny, resolution, init_val=0.0):
        self.center_px_ = center_px  # 栅格中心x轴坐标
        self.center_py_ = center_py  # 栅格中心y轴坐标
        self.width_nx_ = width_nx  # 栅格在x轴的个数
        self.width_ny_ = width_ny  # 栅格在y轴的个数
        self.resolution_ = resolution  # 栅格的分辨率
        self.min_px_ = center_px - width_nx * resolution * 0.5  # 栅格在x轴的最小坐标
        self.min_py_ = center_py - width_ny * resolution * 0.5  # 栅格在y轴的最小坐标
        self.ndata_ =  self.width_nx_ * self.width_ny_  # 栅格地图数据的总长度
        self.data_ = [init_val] * self.ndata_  # 栅格地图数据
    
    # 坐标与栅格转换
    def posToGridMapIndex(self, pos, min_pos, max_index):
        """ 将单个坐标信息转化为栅格的下标
            :param pos: 输入坐标
            :param min_pos: 最小坐标
            :param max_index: 最大下标
        """

        index = int(np.floor((pos - min_pos) / self.resolution_))
        if index >= 0 and index < max_index:
            return index
        else:
            return None 
    
    def pointPosToGridMapIndex(self, pos_x, pos_y):
        """ 将坐标转化为对应栅格
            :param pos_x: x轴坐标
            :param pos_y: y轴坐标
        """

        index_x = self.posToGridMapIndex(pos_x, self.min_px_, self.width_nx_)
        index_y = self.posToGridMapIndex(pos_y, self.min_py_, self.width_ny_)
        return index_x, index_y
    
    def gridMapIndexToPos(self, index, min_pos):
        """ 将栅格单个下标转化为栅格中心坐标
            :param index: 栅格单个下标
            :param min_pos: 栅格最小坐标
        """

        return index * self.resolution_ + min_pos + self.resolution_ * 0.5
    
    def gridMapIndexToPointPos(self, index_x, index_y):
        """ 将栅格转化为栅格中心坐标
            :param index_x: 栅格x轴下标
            :param index_y: 栅格y轴下标
        """ 

        pos_x = self.gridMapIndexToPos(index_x, self.min_px_)
        pos_y = self.gridMapIndexToPos(index_y, self.min_py_)
        return pos_x, pos_y
    
    def gridMapIndexToDataIndex(self, index_x, index_y):
        """ 将栅格下标转化为数据下标
            :param index_x: 栅格x轴下标
            :param index_y: 栅格y轴下标
        """

        return index_x + index_y * self.width_nx_
    
    # 读取和修改栅格地图
    def getGridMapValueByIndex(self, index_x, index_y):
        """ 根据给出的栅格下标读取栅格地图
            :param index_x: 栅格x轴下标
            :param index_y: 栅格y轴下标
        """

        data_index = self.gridMapIndexToDataIndex(index_x, index_y)
        if data_index >=0 and data_index < self.ndata_:
            return self.data_[data_index]
        else:
            return None
    
    def setGridMapValueByIndex(self, index_x, index_y, value):
        """ 根据给出的栅格下标修改栅格地图
            :param index_x: 栅格x轴下标
            :param index_y: 栅格y轴下标
            :param value: 修改的值
        """

        # 首先确定输入的是有效值
        if (index_x is None) or (index_y is None):
            print(index_x, index_y)
            return False, False
        # 修改对应值
        data_index = self.gridMapIndexToDataIndex(index_x, index_y)
        if data_index >=0 and data_index < self.ndata_:
            self.data_[data_index] = value
            return True
        else:
            return False

    def setGridMapValueByPos(self, pos_x, pos_y, value):
        """ 根据给出的坐标点修改栅格地图
            :param pos_x: x轴坐标
            :param pos_y: y轴坐标
            :param value: 修改的值
        """

        # 首先找到坐标对应的栅格下标
        index_x, index_y = self.pointPosToGridMapIndex(pos_x, pos_y)
        # 验证转换是否有效
        if (index_x is None) or (index_y is None):
            return False
        # 设置值
        return self.setGridMapValueByIndex(index_x, index_y, value)
    
    def setGridMapbyPolyon(self, polyon_x, polyon_y, value, inside=True):
        """ 根据给出的多边形对栅格地图进行修改
            :param polyon_x: 多边形的横坐标
            :param polyon_y： 多边形的纵坐标
            :param value: 要修改的值
            :param inside: 修改的是多边形内还是多边形外
        """

        # 首先判断多边形是否闭合
        if polyon_x[0] != polyon_x[-1] or polyon_y[0] != polyon_y[-1]:
            # 如果多边形非闭合，则将其进行闭合
            polyon_x.append(polyon_x[0])
            polyon_y.append(polyon_y[0])
        # 开始遍历每一个栅格中心是否在多边形内
        for index_x in range(0, self.width_nx_):
            for index_y in range(0, self.width_ny_):
                # 计算栅格中心坐标
                pos_x, pos_y = self.gridMapIndexToPointPos(index_x, index_y)
                # 判断点是否在多边形内
                is_inside = self.isInPolyon(polyon_x, polyon_y, pos_x, pos_y)
                if is_inside is inside:
                    self.setGridMapValueByIndex(index_x, index_y, value)
    
    @staticmethod
    def isInPolyon(polyon_x, polyon_y, pos_x, pos_y):
        """ 判断点是否在多边形内
            采用的方法是射线法，从当前引出任意一条射线，如果与多边形交点个数为奇数个则点在多边形内，反之点不在多边形内
            :param polyon_x: 多边形的x轴坐标
            :param polyon_y: 多边形的y轴坐标
            :param pos_x: 点的x坐标
            :param pos_y: 点的y坐标
        """

        # 遍历多边形每一条边, 多边形必须是闭合的（即第一个点就是最后一个点）
        is_inside = False
        for current_point_index in range(0, len(polyon_x) - 1):
            neighbor_point_index = (current_point_index + 1) % len(polyon_x)
            # 判断pos_x是不是在两个点的x坐标之间
            if pos_x > min(polyon_x[current_point_index], polyon_x[neighbor_point_index]) and pos_x < max(polyon_x[current_point_index], polyon_x[neighbor_point_index]):
                # 在之间
                # 判断射线是否相交
                if (polyon_y[neighbor_point_index] - polyon_y[current_point_index]) / (polyon_x[neighbor_point_index] - polyon_x[current_point_index]) * (pos_x - polyon_x[current_point_index]) + polyon_y[current_point_index] - pos_y > 0.0:
                    # 相交
                    is_inside = not is_inside
            else:
                # 不在之间
                continue
        return is_inside
    
    def checkOccupationByIndex(self, index_x, index_y, occupied_value=1.0):
        """ 判断栅格地图对应下标位置的值是否大于occupied_value
            :param index_x: 栅格x轴下标
            :param index_y: 栅格y轴下标
            :param occupied_value: 比较的值
        """

        value = self.getGridMapValueByIndex(index_x, index_y)
        if value >= occupied_value:
            return True
        else:
            return False
    
    def expandGrid(self):
        """ 扩展栅格地图中的有效值范围
        """
        
        # 遍历栅格，判断栅格是否有效
        checked_index_x, checked_index_y = [], [] 
        for index_x in range(0, self.width_nx_):
            for index_y in range(0, self.width_ny_):
                if self.checkOccupationByIndex(index_x, index_y):
                    checked_index_x.append(index_x)
                    checked_index_y.append(index_y)

        # 将有效栅格的8邻域进行有效化
        for index_x, index_y in zip(checked_index_x, checked_index_y):
            self.setGridMapValueByIndex(index_x, index_y + 1, 1.0)
            self.setGridMapValueByIndex(index_x + 1, index_y, 1.0)
            self.setGridMapValueByIndex(index_x, index_y - 1, 1.0)
            self.setGridMapValueByIndex(index_x - 1, index_y, 1.0)
            self.setGridMapValueByIndex(index_x + 1, index_y + 1, 1.0)
            # self.setGridMapValueByIndex(index_x + 1, index_y - 1, 1.0)
            self.setGridMapValueByIndex(index_x - 1, index_y - 1, 1.0)
            # self.setGridMapValueByIndex(index_x - 1, index_y + 1, 1.0)

    # 进行可视化
    def plotGridMap(self, ax=None):
        """ 将栅格地图进行可视化
        """

        visual_data = np.reshape(np.array(self.data_), (self.width_ny_, self.width_nx_))
        if not ax:
            fig, ax = plt.subplots()
        heat_map = ax.pcolor(visual_data, cmap="Blues", vmin=0.0, vmax=1.0)
        plt.axis("equal")
        # plt.show()

        return heat_map

# 测试函数
def testPositionSet():
    grid_map = GridMap(10.0, -0.5, 100, 120, 0.5)

    grid_map.setGridMapValueByPos(10.1, -1.1, 1.0)
    grid_map.setGridMapValueByPos(10.1, -0.1, 1.0)
    grid_map.setGridMapValueByPos(10.1, 1.1, 1.0)
    grid_map.setGridMapValueByPos(11.1, 0.1, 1.0)
    grid_map.setGridMapValueByPos(10.1, 0.1, 1.0)
    grid_map.setGridMapValueByPos(9.1, 0.1, 1.0)

    grid_map.plotGridMap()

# 测试函数
def testPolygonSet():
    ox = [0.0, 20.0, 50.0, 100.0, 130.0, 40.0]
    oy = [0.0, -20.0, 0.0, 30.0, 60.0, 80.0]

    grid_map = GridMap(60.0, 30.5, 600, 290, 0.7)

    grid_map.setGridMapbyPolyon(ox, oy, 1.0, inside=False)

    grid_map.plotGridMap()

    plt.axis("equal")
    plt.grid(True)

# 主函数
def main():
    print("Test grid_map start")
    testPositionSet()
    testPolygonSet()
    plt.show()
    print("Done")

if __name__ == "__main__":
    main()