# _*_ coding:utf-8 _*_

import tushare as ts
import pandas as pd
import math
import matplotlib.pyplot as plt
from DataProcessing.filter_extreme import Filter_extreme
from DataProcessing.standardize import Standardize
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

class SingleFactor():
    """
    单因子选股的策略

    策略：
        数据选取部分：
            1. 选取的数据集是 上证50大盘股
            2. 特征值是当月月底的市盈率，目标值是下月月初到月底的收益率
        算法选择：
            1. 采用正规方程（样本小于100k）
            2. 采用岭回归进行拟合，因为其包含正则化
    """

    def __init__(self):
        self.token = "b655a4da4dbac0ae19047a34bf0f0be00f52dafa9dc92529c929b55c"
        self.pro = ts.pro_api(self.token)
        self.filter_extreme = Filter_extreme()
        self.standard = Standardize()


    def getSingalData(self, stock_code):
        """
        获取单只股票的特征值，这里的特征项是 市盈率 (TTM)的倒数 作为 特征值 EP
        :return:
        """
        # 2018 - 2019 每月最后一天交易日是多少
        month_end_tradeDate = ['20171229',
                               '20180131', '20180228', '20180330', '20180427', '20180531', '20180629',
                               '20180731', '20180831', '20180928', '20181031', '20181130', '20181228',
                               '20190131', '20190228', '20190329', '20190430', '20190531', '20190628',
                               '20190731', '20190830', '20190930']

        data = pd.DataFrame()
        for trade_date in month_end_tradeDate:
            df = self.pro.daily_basic(ts_code=stock_code, trade_date=trade_date,
                                 fields='pe_ttm')
            data = pd.concat([data, df])

        data.index = range(0, len(data))
        data['EP'] = 1 / data['pe_ttm']

        return data

        # 将data部分用plot展示下
        # 中位数法去极值
        # self.filter_extreme = Filter_extreme()
        # fig = plt.figure(figsize=(20, 8))
        # ax = data['EP'].plot.kde(label='EP')
        # ax = self.filter_extreme.filter_extreme_MAD(data['EP'], 2).plot.kde(label='MAD')
        # ax = self.filter_extreme.filter_extreme_3sigma(data['EP'], 2).plot.kde(label='3sigma')
        # ax = self.filter_extreme.filter_extreme_percentile(data['EP']).plot.kde(label='percent')
        # ax.legend()
        # plt.show()


    def getSingalTarget(self, stock_code):
        """
        获取单只股票的数据
        :param stock_code:
        :return:
        """
        target_data = self.pro.monthly(ts_code=stock_code, start_date='20180101', end_date='20191101',
                             fields='ts_code,trade_date,open,close')

        # 进行倒叙
        target_data = target_data.sort_values(by=['trade_date'], ascending=[True])
        target_data.index = range(0, len(target_data))

        # 计算收益率，进行数据回滚，并处理回滚带来的空值
        target_data['close_pro'] = target_data['close'].shift(1)

        if math.isnan(target_data['close_pro'][0]):
            target_data.iloc[[0], [4]] = target_data['open'][0]

        target_data['ratio'] = round((target_data['close'] / target_data['close_pro'] - 1) * 100, 2)

        # 只传递相关的属性
        target_data = target_data[['trade_date', 'ratio']].copy()

        return target_data


    def LinearRegression(self):
        """
        单因子线性回归
        :return:
        """
        # 1 数据获取  获取股票 '600000.SH' 的特征值和目标值
        stock_code = '600000.SH'
        data = self.getSingalData(stock_code)
        target = self.getSingalTarget(stock_code)

        # 2 特征提取
        data = data['EP']
        target = target['ratio']

        # 3 数据转化(对目标值以及)
        data = self.filter_extreme.filter_extreme_3sigma(data)
        target = self.filter_extreme.filter_extreme_3sigma(target)

        # 4 数据集划分
        x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25)

        x_train = x_train.values.reshape(-1, 1)
        x_test = x_test.values.reshape(-1, 1)
        y_train = y_train.values.reshape(-1, 1)
        y_test = y_test.values.reshape(-1, 1)

        # 5 进行标准化处理
        self.transfor_x = StandardScaler()    # 创造一个转化器
        x_train = self.transfor_x.fit_transform(x_train)
        x_test = self.transfor_x.transform(x_test)

        self.transfor_y = StandardScaler()
        y_train = self.transfor_y.fit_transform(y_train)
        y_test = self.transfor_y.transform(y_test)

        # 4 模型训练
        # 4.1 使用正规方程进行模型训练
        self.model_LinearRegression(x_train, x_test, y_train, y_test)

        # 4.2 使用梯度下降算法进行模型训练
        self.model_SGDRegressor(x_train, x_test, y_train, y_test)

        # 4.3 使用岭回归进行模型训练
        self.model_Ridge(x_train, x_test, y_train, y_test)


    def model_LinearRegression(self, x_train, x_test, y_train, y_test):
        """
        正规化方程求解
        :param x_train:
        :param x_test:
        :param y_train:
        :param y_test:
        :return:
        """
        # 创建一个估计器
        estimator = LinearRegression()

        # 传入训练集进行机器学习
        estimator.fit(x_train, y_train)

        # 打印模型 结果系数 和 偏置
        print('模型结果系数为: ', estimator.coef_, '模型偏置为: ', estimator.intercept_)

        # 保存模型结果
        # joblib.dump(estimator, '../tmp/model_LinearRegression.pkl')
        # 加载模型结果
        # model = joblib.load('../tmp/model_LinearRegression.pkl')
        # y_predict = model.predict(x_test)

        # 模型评估
        y_predict = estimator.predict(x_test)
        print(y_predict)

        error = mean_squared_error(y_test, y_predict)
        print('正规方程优化的均方误差为:\n', error)

        # y_predict_prior = self.transfor_y.inverse_transform(y_predict)
        # y_test_prior = self.transfor_y.inverse_transform(y_test)


    def model_SGDRegressor(self, x_train, x_test, y_train, y_test):
        """
        梯度下降进行模型训练
        :param x_train:
        :param x_test:
        :param y_train:
        :param y_test:
        :return:
        """
        estimator = SGDRegressor(learning_rate='constant', eta0=0.01)

        estimator.fit(x_train, y_train)

        print('模型结果系数为: ', estimator.coef_, '模型偏置为: ', estimator.intercept_)

        y_predict = estimator.predict(x_test)

        error = mean_squared_error(y_test, y_predict)

        print('梯度下降优化的均方误差为:\n', error)


    def model_Ridge(self, x_train, x_test, y_train, y_test):
        """
        岭回归进行模型训练
        :param x_train:
        :param x_test:
        :param y_train:
        :param y_test:
        :return:
        """
        estimator = Ridge(alpha=1.0)

        estimator.fit(x_train, y_train)

        print('模型结果系数为: ', estimator.coef_, '模型偏置为: ', estimator.intercept_)

        y_predict = estimator.predict(x_test)

        error = mean_squared_error(y_test, y_predict)

        print('岭回归优化的均方误差为:\n', error)


if __name__ == "__main__":
    singleFact = SingleFactor()
    singleFact.LinearRegression()



