# _*_ coding:utf-8 _*_

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


class Logistic_regression():
    """
    逻辑回归
    解决二分类问题的利器
    """
    def __init__(self):
        pass

    def logistic_regression(self):
        """
        逻辑回归问题
        :return:
        """
        # 获取数据
        columns = ['sample code number',
                   'Clump Thickness',
                   'Uniformity of Cell Size',
                   'Uniformity of Cell Shape',
                   'Marginal Adhesion',
                   'Single Bpithlial Cell Size',
                   'Bare Nuclei',
                   'Bland Chromation',
                   'Normal Nucleoli',
                   'Mitoses',
                   'Class']

        data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', names=columns)

        # 数据处理， 有些值存在 "?" 问题，要用空值进行替代
        data = data.replace(to_replace='?', value=np.nan)
        # 删除包含缺失值的行， 这里用dropna, axis=0, 表示删除行， axis=1，表示删除列， 默认的是axis=0,  drop是删除制定的行和列
        data = data.dropna()

        # 进行数据分割
        x_train, x_test, y_train, y_test = train_test_split(data[columns[1:10]], data[columns[10]], test_size=0.25)

        # 进行标准化处理
        std = StandardScaler()
        x_train = std.fit_transform(x_train)
        x_test = std.transform(x_test)

        # 模型训练
        estimator = LogisticRegression(C=1.0)
        estimator.fit(x_train, y_train)
        # print(estimator.coef_)
        # print(estimator.intercept_)

        # 模型评估
        y_predict = estimator.predict(x_test)

        # 准确率 和 召回率(主要看恶性肿瘤的召回率，如果召回率高，说明模型越有效)
        model_score = estimator.score(x_test, y_test)
        print('准确率是: ', model_score)
        model_report = classification_report(y_test, y_predict, labels=[2, 4], target_names=['良性', '恶性'])
        print('召回率是: ', model_report)


if __name__ == '__main__':
    logistic = Logistic_regression()
    logistic.logistic_regression()