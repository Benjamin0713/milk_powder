import shap
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import xgboost
from sklearn.model_selection import train_test_split
#加载可视化
shap.initjs()
#模型model
# data = pd.read_csv('./data/Mean_Cope2.csv', header=0, encoding='utf-8')
#
# y = data['o1_r'].values
# data = data.drop(columns=['o1_r', 'o2_r', 'o3_r', 'o4_r', 'o5_r', 'o6_r', 'o7_r', 'o8_r', 'o9_r', 'o10_r', 'o11_r', 'o12_r'])
# # 特征值
# X = data.values
#
# X_test = X
# y_test = y
#
# # params = {
# #     'task': 'train',
# #     'boosting_type': 'gbdt',  # 设置提升类型
# #     'objective': 'regression',  # 目标函数
# #     'metric': {'l2', 'auc'},  # 评估函数
# #     'num_leaves': 31,  # 叶子节点数
# #     'learning_rate': 0.05,  # 学习速率
# #     'feature_fraction': 0.9,  # 建树的特征选择比例
# #     'bagging_fraction': 0.8,  # 建树的样本采样比例
# #     'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
# #     'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
# # }
# gbm = lgb.LGBMRegressor(objective='regression',
#                               max_depth = 3,
#                               learning_rate=0.1, n_estimators=30,
#                               metric='mae', bagging_fraction = 0.8,feature_fraction = 0.8)
# gbm.fit(X, y, eval_set=[(X_test, y_test)], eval_metric='l2', early_stopping_rounds=10)
#
# y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
#
# explainer = shap.TreeExplainer(gbm)
# shap_values = explainer.shap_values(X)
#
# # 可视化第一个prediction的解释   如果不想用JS,传入matplotlib=True
# # shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])
#
# # y_base = explainer.expected_value
# # print(y_base)
# #
# # pred = gbm.predict(gbm.DMatrix(X))
# # print(pred.mean())
# # shap.force_plot(explainer.expected_value, shap_values, X)
#
# # summarize the effects of all the features
# shap.summary_plot(shap_values, X)
# # shap.summary_plot(shap_values, X, plot_type="bar")
# # shap_interaction_values = explainer.shap_interaction_values(X)
# # shap.summary_plot(shap_interaction_values, X)
# # shap.dependence_plot("RM", shap_values, X)
csv_file = './data/Mean_Cope2.csv'

X = pd.read_csv(csv_file,
                          header=0, encoding='utf-8',
                          usecols=[0,1,2,3,4,5,6,7,8,9,10,
                                11,12,13,14,15,16,17,18,19,20
                                ,21,22,23,24,25,26,27,28,29])
y = pd.read_csv(csv_file, header=0, encoding='utf-8',usecols=[30])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from xgboost import XGBRegressor

# 1
# my_model = XGBRegressor()
# # Add silent=True to avoid printing out updates with each cycle
# my_model.fit(X_train, y_train, verbose=False)

#2
# my_model = XGBRegressor(n_estimators=1000)
# my_model.fit(X_train, y_train, early_stopping_rounds=5,
#              eval_set=[(X_test, y_test)], verbose=False)

#3
my_model = XGBRegressor(objective ='reg:squarederror',max_depth = 3,n_estimators=100, learning_rate=0.1,eval_metric='mae',
                        colsample_bytree=0.8,subsample=0.8)
my_model.fit(X_train, y_train, early_stopping_rounds=10,
             eval_set=[(X_test, y_test)],verbose=True)

# my_model.fit(X_train, y_train, verbose=True)
y_pred = my_model.predict(X_test)

explainer = shap.TreeExplainer(my_model)
shap_values = explainer.shap_values(X)  # 传入特征矩阵X，计算SHAP值

# 可视化第一个prediction的解释   如果不想用JS,传入matplotlib=True
shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])

# y_base = explainer.expected_value
# print(y_base)
#
# pred = my_model.predict(xgboost.DMatrix(X))
# print(pred.mean())
# summarize the effects of all the features
shap.summary_plot(shap_values, X)
shap.summary_plot(shap_values, X, plot_type="bar")
# shap_interaction_values = explainer.shap_interaction_values(X)
# shap.summary_plot(shap_interaction_values, X)