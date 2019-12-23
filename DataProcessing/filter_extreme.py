# _*_ coding:utf-8 _*_

import numpy as np


class  Filter_extreme():
    """
    去极值处理，有如下三种常见方法：
        1. 绝对值差中位数法 （MAD）
        2. 3σ 法又称为标准差法
        3. 百分位法
    """
    def __init__(self):
        pass


    def filter_extreme_MAD(self, series, n):
        """
        绝对值差中位数法
        逻辑：
            第一步，找出所有因子的中位数 Xmedian
            第二步：得到每个因子与中位数的绝对偏差值 Xi−Xmedian
            第三步：得到绝对偏差值的中位数 MAD
            第四步：确定参数 n ，从而确定合理的范围为 [Xmedian − nMAD,Xmedian + nMAD]
        :return:
        """
        # 找出其中的中位数
        median = series.quantile(0.5)

        # 找到每个因子与中位数的绝对值偏差值
        new_median = ((series - median).abs()).quantile(0.50)

        # 确定上线限，即确定合理的范围
        max_range = median + n * new_median
        min_range = median - n * new_median

        return np.clip(series, min_range, max_range)


    def filter_extreme_3sigma(self, series, n=3):
        """
        3σ 法又称为标准差法
        逻辑：
            第一步：计算出因子的平均值与标准差
            第二步：确认参数 n         （这里选定 n = 3 ）
            第三步：确认因子值的合理范围为 [Xmean−nσ,Xmean nσ]
        :return:
        """
        mean = series.mean()
        std = series.std()
        max_range = mean + n * std
        min_range = mean - n * std

        return np.clip(series, min_range, max_range)


    def filter_extreme_percentile(self, series, min = 0.10, max = 0.90):
        """
        百分位法
        逻辑：
            将因子值进行升序的排序，对排位百分位高于 97.5%或排位百分位低于2.5% 的因子值，进行类似于 MAD  、 3σ  的方法进行调整
        :return:
        """
        series = series.sort_values()

        q = series.quantile([min, max])

        return np.clip(series, q.iloc[0], q.iloc[1])


if __name__ == '__main__':
    filter = Filter_extreme()