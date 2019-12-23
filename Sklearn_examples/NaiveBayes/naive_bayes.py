# _*_ coding:utf-8 _*_

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


class Naive_Bayes():
    """
    朴素贝叶斯
    """
    def __init__(self):
        pass

    def naiveBayes(self):
        """
        朴素贝叶斯算法，带拉普拉斯平滑
        :return:
        """
        # 获取数据
        iris = load_iris()

        # 数据分割
        x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25)

        # 模型训练
        estimator = MultinomialNB()
        estimator.fit(x_train, y_train)
        y_predict = estimator.predict(x_test)

        scores = estimator.score(x_test, y_test)
        print('准确率是: ', scores)

        reports = classification_report(y_test, y_predict, labels=[0, 1, 2], target_names=['setosa', 'versicolor', 'virginica'])
        print("召回率是: ", reports)


if __name__ == "__main__":
    naiveBayes = Naive_Bayes()
    naiveBayes.naiveBayes()