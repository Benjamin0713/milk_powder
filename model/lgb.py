import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_absolute_percentage_error

csv_file = '../data/Mean_Cope2.csv'

feature = pd.read_csv(csv_file,
                          header=0, encoding='utf-8',
                          usecols=[0,1,2,3,4,5,6,7,8,9,10,
                                11,12,13,14,15,16,17,18,19,20
                                ,21,22,23,24,25,26,27,28,29])
target = pd.read_csv(csv_file, header=0, encoding='utf-8',usecols=[30])

# print(type(feature))
# print(type(target))
# print(feature)
# print(target)
X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.01)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',  # 设置提升类型
    'objective': 'regression',  # 目标函数
    'metric': {'l2', 'mae'},  # 评估函数
    'num_leaves': 31,  # 叶子节点数
    'learning_rate': 0.1,  # 学习速率
    'feature_fraction': 0.8,  # 建树的特征选择比例
    'bagging_fraction': 0.8,  # 建树的样本采样比例
    'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
    'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
}

gbm = lgb.train(params, lgb_train, num_boost_round=200, valid_sets=lgb_eval, early_stopping_rounds=50)

y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
print('The MAPE of prediction is:',mean_absolute_percentage_error(y_test, y_pred))