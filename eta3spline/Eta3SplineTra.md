Eta^3 spline trajectory是在Eta^3 spline path的基础上给每一个点加上速度和角速度，相当于在之前规划的路径基础之上进行速度规划，最终得到轨迹。
速度规划的思路相对简单，只考虑以最大加速度达到最大速度，并在终点时达到速度和加速度同时为0即可。其分为了七个阶段：第一个阶段是变加速运动，即加速度以最大变化率达到最大加速度；第二阶段是匀加速运动，速度以最大加速度达到特定速度；第三阶段变加速运动，加速度以最大变化率变为0，并与此同时，速度由特定速度达到最大速度；第四阶段匀加速运动，以最大速度行驶；第五阶段变加速运动加速度以最大变化率达到最大减速度；第六阶段匀减速运动，以最大减速度减速到特定速度；第七阶段，变加速运动加速度以最大变化率达到0，并与此同时，速度也从特定速度变化为0。
以上过程必须确定最大速度是否能够达到，确定的方法没能找到相关资料，需要进一步思考。
在完成轨迹的速度属性定义后，下一步是如何得到时间t所对应的位置，朝向和速度信息。首先，可以判断t时间走过的路程s，这个可以通过之前得到的速度属性计算出来，下一步就是如何通过s得到对应的参数u。首先根据s可以知道特定点是处于哪一个曲线段，而由于对于特定曲线段，s(u)的方程是已知的，而且s(u)的导数也是已知的，因此可以通过牛顿迭代的方法找到特定s对应的u。（这里的迭代初始值十分重要，因为定义域只在[0,1]之间）得到u后，就可以通过eta^3 spline的方程直接得到t时刻对应的位置，一阶导数和二阶导数，也就可以进一步通过微分几何推导得到角速度的值。