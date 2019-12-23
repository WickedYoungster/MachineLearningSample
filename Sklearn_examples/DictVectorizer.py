from sklearn.feature_extraction import DictVectorizer

class dictVectorizer():
    """字典特征抽取，这里的返回是 one-hot 编码"""
    def __init__(self):
        """init model"""
        pass


    def run(self):
        """run model"""

        # 变量声明与实现
        data_value = [{"city": "ShangHai", "tempreture": 10},
                {"city": "BeiJing", "tempreture": 8},
                {"city": "HeFei", "tempreture": 5}]

        # 实例化
        dict = DictVectorizer(sparse=False)

        # 调用fit_transform方法，然后得到转化后的数据格式
        data = dict.fit_transform(data_value)

        # 获取属性名
        featureName = dict.get_feature_names()
        print(featureName)

        # 返回数据之前的格式
        data_inverse = dict.inverse_transform(data)
        print(data_inverse)

        # 返回转换后的数据格式
        print(data)


if __name__ == "__main__":
    dictVect = dictVectorizer()
    dictVect.run()