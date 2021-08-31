import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import mean_absolute_percentage_error
import lightgbm as lgb
from sklearn.model_selection import train_test_split

csv_file = '../data/Mean_Cope2.csv'

feature = pd.read_csv(csv_file,
                          header=0, encoding='utf-8',
                          usecols=[0,1,2,3,4,5,6,7,8,9,10,
                                11,12,13,14,15,16,17,18,19,20
                                ,21,22,23,24,25,26,27,28,29])
target = pd.read_csv(csv_file, header=0, encoding='utf-8',usecols=[30])

X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2)
# 1
gbm = lgb.LGBMRegressor(objective='regression',
                              max_depth = 3,
                              learning_rate=0.1, n_estimators=30,
                              metric='rmse', bagging_fraction = 0.8,feature_fraction = 0.8)
gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='l1', early_stopping_rounds=10)

y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)

print('The MAPE of prediction is:', mean_absolute_percentage_error(y_test, y_pred))

#
# print('Feature importances:', list(gbm.feature_importances_))
#
# from sklearn.model_selection import GridSearchCV
# estimator = lgb.LGBMRegressor(num_leaves=31)
# param_grid = {
#     'learning_rate': [0.01, 0.1, 1],
#     'n_estimators': [20, 40]
# }
# gbm = GridSearchCV(estimator, param_grid)
# gbm.fit(X_train, y_train)
# print('Best parameters found by grid search are:', gbm.best_params_)

# 2
# model=lgb.LGBMRegressor(
#         n_estimators=200,
#         learning_rate=0.03,
#         num_leaves=32,
#         colsample_bytree=0.9497036,
#         subsample=0.8715623,
#         max_depth=8,
#         reg_alpha=0.04,
#         reg_lambda=0.073,
#         min_split_gain=0.0222415,
#         min_child_weight=40)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test).clip(0., 20.)
# print('The MAPE of prediction is:', mean_absolute_percentage_error(y_test, y_pred))