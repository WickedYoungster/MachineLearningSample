# _*_ coding:utf-8 _*_
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import export_graphviz
import graphviz

class Tree_model():
    """
    决策树算法实现
    """

    def __init__(self):
        pass


    def tree_algrothim(self):
        """
        决策树具体实现
        :return:
        """
        # 获取数据
        wine = load_wine()
        print(wine)
        # 切分数据
        x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.25)

        # 模型训练
        estimator = DecisionTreeClassifier()
        estimator.fit(x_train, y_train)

        y_predict = estimator.predict(x_test)

        # 准确率和召回率
        score = estimator.score(x_test, y_test)
        print('准确率是:', score)
        report = classification_report(y_test, y_predict, labels=[0, 1, 2], target_names=wine.target_names)
        print('召回率是:', report)

        feature_name = ['酒精', '苹果酸', '灰', '灰的碱性', '镁', '总酚', '类黄酮', '非黄烷类酚类', '花青素', '颜色强度', '色调', 'od280/od315稀释葡萄酒',
                        '脯氨酸']

        dot_data = export_graphviz(estimator
                                        , out_file=None
                                        , feature_names=feature_name
                                        , class_names=["琴酒", "雪莉", "贝尔摩德"]
                                        , filled=True
                                        , rounded=True
                                        )
        graph = graphviz.Source(dot_data)

        with open("./decisiontree.dot", "w") as wFile:
            export_graphviz(estimator, out_file=wFile, feature_names=feature_name)




if __name__ == '__main__':
    tree_model = Tree_model()
    tree_model.tree_algrothim()