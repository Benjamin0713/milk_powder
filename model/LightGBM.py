import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from lightgbm.sklearn import LGBMRegressor

import lightgbm as lgb

data = pd.read_csv('../data/Mean_Cope2.csv', header=0, encoding='utf-8')

y = data['o1_r'].values
data = data.drop(columns=['o1_r', 'o2_r', 'o3_r', 'o4_r', 'o5_r', 'o6_r', 'o7_r', 'o8_r', 'o9_r', 'o10_r', 'o11_r', 'o12_r'])
# 特征值
X = data.values

X_test = X
y_test = y

# params = {
#     'task': 'train',
#     'boosting_type': 'gbdt',  # 设置提升类型
#     'objective': 'regression',  # 目标函数
#     'metric': {'l2', 'auc'},  # 评估函数
#     'num_leaves': 31,  # 叶子节点数
#     'learning_rate': 0.05,  # 学习速率
#     'feature_fraction': 0.9,  # 建树的特征选择比例
#     'bagging_fraction': 0.8,  # 建树的样本采样比例
#     'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
#     'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
# }
gbm = lgb.LGBMRegressor(objective='regression',
                              max_depth = 3,
                              learning_rate=0.1, n_estimators=30,
                              metric='mae', bagging_fraction = 0.8,feature_fraction = 0.8)
gbm.fit(X, y, eval_set=[(X_test, y_test)], eval_metric='l2', early_stopping_rounds=10)

y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)

print('The MAPE of prediction is:', mean_absolute_percentage_error(y, y_pred))