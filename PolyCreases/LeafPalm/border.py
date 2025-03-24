import random
import numpy as np
from matplotlib import pyplot as plt
from shapely import Polygon


class border:
    def __init__(self, param, lower_bound, upper_bound):
        self.param = param  # param中保存2阶多项式参数,即数列中保存三个数
        self.upper = upper_bound  # 边界在x轴上的上界
        self.lower = lower_bound  # 边界在x轴上的下界

    def backbone_fundation(self, x):  # 多项式方程，输入x，返回y
        y = sum(param * x ** (len(self.param) - 1 - idx) for idx, param in enumerate(self.param))
        return y

    def calculate_tangent_angle(self, x_val):
        slope = 0
        if x_val == 0:
            slope = self.param[-2]
        else:
            for idx, p in enumerate(self.param):
                slope += (len(self.param) - 1 - idx) * p * x_val ** (len(self.param) - 2 - idx)
        # 计算切线方向向量
        tangent_vector = np.array([1, slope], dtype=float)
        # 计算垂直方向向量
        vertical_vector = np.array([0, 1], dtype=float)

        # 计算切线方向与垂直方向的夹角（弧度）
        angle_rad = np.arctan2(tangent_vector[1], tangent_vector[0]) - np.arctan2(vertical_vector[1],
                                                                                  vertical_vector[0])
        return angle_rad

    def border_define(self, step=0.05, is_branch=False):  # 定义多边形边界
        # step表示多远进行一次边界点设置
        x = self.lower
        point_set = []  # 记录边界上的点集
        while x <= self.upper:
            y = self.backbone_fundation(x)
            # if is_branch:
            #     bias = 0.1
            # else:
            bias = 0.08 / np.sin(self.calculate_tangent_angle(x))  # np.sin函数的参数是弧度，而不是角度
            point_set.append((x, y + bias))
            point_set.append((x, y - bias))
            x = round(x + step, 2)
        border = [0] * len(point_set)  # 将点集重新排列，形成边界
        i = 0
        while i < len(point_set) / 2:
            border[i] = point_set[2 * i]
            border[-i - 1] = point_set[2 * i + 1]
            i = i + 1
        return border


if __name__ == '__main__':
    border_creater_1 = border(
        [-2.309277599186426, -3.9903253136852856, -2.3730508002504127, -2.4450928187126286, -0.8991630453268774],
        -0.92708023, 0.5134968)
    border_1 = border_creater_1.border_define()
    border_1 = [((x / 2.0) + 0.5, (y / 2.0) + 0.5) for x, y in border_1]

    p = Polygon(border_1)
    x, y = zip(*p.exterior.coords)
    plt.plot(x, y, label='area', color='blue')
    x, y = zip(*border_1)
    plt.scatter(x, y, c='blue')

    border_creater_2 = border(
        [0.19041597686984477, 0.18299145114394438, -0.28130344560471743, -0.42697657441020814, 0.14577144669409092],
        -1.0283488, 0.45339728)
    border_2 = border_creater_2.border_define()
    border_2 = [((x / 2.0) + 0.5, (y / 2.0) + 0.5) for x, y in border_2]
    p = Polygon(border_2)
    x, y = zip(*p.exterior.coords)
    plt.plot(x, y, label='area', color='blue')
    x, y = zip(*border_2)
    plt.scatter(x, y, c='blue')

    border_creater_3 = border(
        [-0.20206152840245054, 0.017435304668667377, 0.516291273609968, -0.5630598715774259, 0.4606384049791271],
        -0.30461669, 0.88865701)
    border_3 = border_creater_3.border_define()
    border_3 = [((x / 2.0) + 0.5, (y / 2.0) + 0.5) for x, y in border_3]
    p = Polygon(border_3)
    x, y = zip(*p.exterior.coords)
    plt.plot(x, y, label='area', color='blue')
    x, y = zip(*border_3)
    plt.scatter(x, y, c='blue')
    plt.axis('equal')
    plt.show()
