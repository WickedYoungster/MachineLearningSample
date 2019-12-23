# _*_ coding: utf-8 _*_

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

class KNN_Model():
    """
    k-近邻算法
    """
    def __init__(self):
        pass


    def KNN(self):
        """
        这里采用熟悉的鸢尾花
        :return:
        """
        # 获取数据
        lr = load_iris()

        # 数据分割
        x_train, x_test, y_train, y_test = train_test_split(lr.data, lr.target, test_size=0.25)

        # 数据处理
        std = StandardScaler()

        x_train = std.fit_transform(x_train)
        x_test = std.transform(x_test)

        # 模型训练
        estimator = KNeighborsClassifier()
        # estimator.fit(x_train, y_train)
        #
        # y_predict = estimator.predict(x_test)
        # print(y_predict)
        #
        # socre = estimator.score(x_test, y_test)
        # print("准确率是:", socre)

        # 通过网格搜索， n_neighbors 为参数列表
        params = {"n_neighbors": [3, 5, 7]}

        gs = GridSearchCV(estimator, param_grid=params, cv=10)

        gs.fit(x_train, y_train)
        k = gs.best_params_
        y_predict = gs.predict(x_test)
        print(k)
        print(gs.best_estimator_)
        print(gs.score(x_test, y_test))

        reports = classification_report(y_test, y_predict, target_names=lr.target_names)
        print(reports)





if __name__ == '__main__':
    knn = KNN_Model()
    knn.KNN()