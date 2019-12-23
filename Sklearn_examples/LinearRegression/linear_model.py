import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib


def mylinear():
    """
    线性回归预测房子价格
    :return:
    """
    # 获取数据
    lb = load_boston()

    # 分割数据集到训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)


    # 进行标准化处理
    # 对特征值和目标值分别进行标准化处理

    # 特征值
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    # 目标值
    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))
    y_test = std_y.transform(y_test.reshape(-1, 1))


    # estimator 预测
    # 基于正规方程求解方式预测结果
    lr = LinearRegression()

    lr.fit(x_train, y_train)

    # 打印特征值前系数
    print(lr.coef_)

    # 预测测试集的房子价格
    y_predict = std_y.inverse_transform(lr.predict(x_test))

    #print(y_predict)

    print("正规方程的均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_predict))

    # 使用梯度下降的方式预测结果
    sgd = SGDRegressor(learning_rate='constant', eta0=0.01)

    sgd.fit(x_train, y_train)

    # 打印梯度下降的特征值前系数
    print(sgd.coef_)

    # 求的梯度下降方式的预测结果
    y_sgd_predict = std_y.inverse_transform(sgd.predict(x_test))

    #print(y_sgd_predict)

    print("梯度下降的均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_sgd_predict))


    # 使用岭回归的方式预测结果
    rd = Ridge(alpha=1)

    rd.fit(x_train, y_train)

    # 打印梯度下降的特征值前系数
    print(rd.coef_)

    # 求的梯度下降方式的预测结果
    y_rd_predict = std_y.inverse_transform(rd.predict(x_test))

    #print(y_rd_predict)

    print("岭回归的均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_rd_predict))

    # 保存训练好的模型
    joblib.dump(rd, "./tmp/test.pkl")

    # 读取保存好的训练模型
    rd_model = joblib.load("./tmp/test.pkl")

    # 预测房价结果
    y_predict = std_y.inverse_transform(rd_model.predict(x_test))
    print("保存模型的预测结果是：", y_predict)

if __name__ == "__main__":
    mylinear()