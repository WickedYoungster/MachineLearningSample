# _*_ coding:utf-8 _*_

import pandas as pd
import math

class Neutralization():
    """
    中性化处理
    中性化目的：在于消除因子中的偏差和不需要的影响
    主要的方法是利用回归得到一个与风险因子线性无关的因子，即通过建立线性回归，提取残差作为中性化后的新因子。
    这样处理后的中性化因子与风险因子之间的相关性严格为零。

    在实际中，两个最典型的中性化例子就是市值中性化和行业中性化。
    比如日成交额这个数据，其实是受市值的影响很大，市值大的股票通常每日的成交额较大，如果同时将成交额和市值两个因子放入模型，就会出现多重共线性问题，无法准确估计出影响力的大小。
    因此，代入模型之前先要对成交额因子提纯，排除市值的影响。具体方法是将成交额和市值进行线性回归，减去市值的影响，剩下的残差就是独立的影响因子。
    同样，行业对因子的影响也很普遍，比如前期提到的股息率因子就受不同行业特征的影响，需要通过行业中性化先排除行业的影响，具体方法是将行业设为虚拟变量，本期这里就不细讲了。
    """

    def __init__(self):
        pass


    # def neutralization(self, factor, mkt_cap=False, industry=True):
    #     """
    #     中性化处理
    #     :param mkt_cap:
    #     :param industry:
    #     :return:
    #     """
    #     y = factor
    #     if type(mkt_cap) == pd.Series:
    #         LnMktCap = mkt_cap.apply(lambda x: math.log(x))
    #         if industry: # 行业、市值
    #             # 获取行业代码标识
    #             dummy_industry = get_industry_exposure(factor.index)
    #             x = pd.concat([LnMktCap, dummy_industry.T], axis=1)
    #
    #         else:  # 仅市值
    #             x = LnMktCap
    #
    #     elif industry: # 仅行业
    #         dummy_industry = get_industry_exposure(factor.index)
    #         x = dummy_industry.T
    #
    #     result = sm.OLS(y.astype(float), x.astype(float)).fit()
    #
    #
    #     return result.resid
