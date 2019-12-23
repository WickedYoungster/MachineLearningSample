from sklearn.feature_extraction.text import CountVectorizer
import jieba

class countVectorizer():
    """

    """


    def __init__(self):
        """init model"""
        pass


    def run(self):
        """run model"""
        ## --------------  英文文本特征化
        # 变量声明
        data_value = ["python is a powerful language", "python can do everything as far as us can image"]

        # 实例化
        countVect = CountVectorizer()

        #抽取特征
        data = countVect.fit_transform(data_value)

        # 获取属性名
        feature_name = countVect.get_feature_names()
        print(feature_name)

        # 利用toarray将 sparse 转化为 array数组
        print(data.toarray())


        ## --------------  中文文本特征化，这里需要 导入jieba库，将中文语句进行分解
        # 分解汉语句子
        con1 = jieba.cut("今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。")

        # 转换成列表
        content1 = list(con1)
        print("content value is:", content1)

        # 把列表转换成字符串
        c1 = ' '.join(content1)
        print("c1 value is:", c1)







if __name__ == "__main__":
    countVect = countVectorizer()
    countVect.run()