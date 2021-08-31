import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, linear_model
from sklearn.linear_model import LinearRegression
# 使用r2_score对模型评估
from sklearn.metrics import mean_squared_error, r2_score
# Window系统下设置字体为SimHei
plt.rcParams['font.sans-serif'] = ['SimHei']
# Mac系统下设置字体为Arial Unicode MS
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

data = pd.read_csv('../data/Mean_Cope2.csv', header=0, encoding='utf-8')
# print(data)
# print(type(data))

# 目标值
y = data['o1_r'].values
data = data.drop(columns=['o1_r', 'o2_r', 'o3_r', 'o4_r', 'o5_r', 'o6_r', 'o7_r', 'o8_r', 'o9_r', 'o10_r', 'o11_r', 'o12_r'])
# 特征值
X = data.values
# print(X)
# print(y)
# 划分训练集与测试集
# X_train, X_test, y_train, y_test = train_test_split(X,
                                                    # y,
                                                    # test_size=0.1,
                                                    # random_state=0)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# print(X_test, X_test, y_train, y_test)

# # 初始化标准化器
# min_max_scaler = preprocessing.MinMaxScaler()
# # 分别对训练和测试数据的特征以及目标值进行标准化处理
# # X_train = min_max_scaler.fit_transform(X_train)
# # y_train = min_max_scaler.fit_transform(y_train.reshape(-1, 1))
# # X_test = min_max_scaler.fit_transform(X_test)
# # y_test = min_max_scaler.fit_transform(y_test.reshape(-1, 1))
# X = min_max_scaler.fit_transform(X)
# y = min_max_scaler.fit_transform(y.reshape(-1, 1))
#
# lr = LinearRegression()
# # 使用训练数据进行参数估计
# # lr.fit(X_train, y_train)
# lr.fit(X, y)
# # 使用测试数据进行回归预测
# # y_test_pred = lr.predict(X_test)
#
# # 训练数据的预测值
# # y_train_pred = lr.predict(X_train)
# # print(y_train_pred)
# y_pred = lr.predict(X)
model_LinearRegression = linear_model.LinearRegression()

model_LinearRegression.fit(X,y)
y_pred = model_LinearRegression.predict(X)
print(y_pred)

# 计算均方差
# train_error = [mean_squared_error(y_train, [np.mean(y_train)] * len(y_train)),
               # mean_squared_error(y_train, y_train_pred)]
# print(train_error)

# 线性回归的系数
# print('线性回归的系数为:\n w = %s \n b = %s' % (lr.coef_, lr.intercept_))

# 平均绝对百分比误差MAPE
print("MAPE: ",sklearn.metrics.mean_absolute_percentage_error(y, y_pred))
