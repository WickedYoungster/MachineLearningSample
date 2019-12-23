# _*_ coding: utf-8 _*_

import math

class Standardize():
    """
    标准化用于多个不同量级指标之间需要互相比较或者数据需要变得集中
    标准化处理：
        1. 对数法（将不同纬度的因子转化到同一个纬度上）：例如对两个因子求对数，相当于比较两者的增长率，增长率的值往往比较小，可以比较。

        2.min-max标准化   （观察值 – 最小值）/（最大值 – 最小值）
        这个方法有点类似之前百分数法，先把这组数据的最大值和最小值之差求出来，然后看观察值超出的最小值的值占大小之差的百分数，
        如果观察值是最大值，则这个值为1，如果是最小值，则为0，也就是所有观测者都会落在[0,1]之间。

        3.z-score  计算观测值和平均值的差是标准差的几倍（除以标准差的原因是 只有以衡量波动的标准差才能确定偏离的幅度是大是小，不然观测值与平均值的差的幅度不能确定偏离程度的大小）
    """

    def __init__(self):
        pass


    def log_standard(self, series):
        """
        对数标准化
        :param series:
        :return:
        """

        return series.apply(lambda x: math.log(x))


    def min_max_standard(self, series):
        """
        min-max 标准法
        （观察值 – 最小值）/（最大值 – 最小值） 所有观测者都会落在[0,1]之间。
        :param series:
        :return:
        """
        max = series.max()
        min = series.min()

        return series.apply(lambda x: (x - min)/(max - min))


    def z_score(self, series):
        """
        z-score 法
        因子研究中最常用的标准化方法
        标准化后的数值 = （观测值 – 平均值）/标准差  （处理后的数据从有量纲转化为无量纲，使得不同的指标能够进行比较和回归）
        :param series:
        :return:
        """
        std = series.std()

        mean = series.mean()

        return (series - mean) / std
