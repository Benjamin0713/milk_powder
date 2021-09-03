import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.metrics
from sklearn import tree
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
# 读取文件

data = pd.read_csv('../data/Mean_Cope2.csv', header=0, encoding='utf-8')
# 目标值
y = np.array(data.get(['o1_r']))
data = data.drop(columns=['o1_r', 'o2_r', 'o3_r', 'o4_r', 'o5_r', 'o6_r', 'o7_r', 'o8_r', 'o9_r', 'o10_r', 'o11_r', 'o12_r'])
# 特征值
X = np.array(data)
# print(X)
# print(y)

# modelling
knr = KNeighborsRegressor()
knr.fit(X,y)

y_pred =knr.predict(X)

# MAPE
print("MAPE: ", sklearn.metrics.mean_absolute_percentage_error(y, y_pred))
