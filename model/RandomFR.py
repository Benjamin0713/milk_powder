import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

csv_file = '../data/Mean_Cope2.csv'

feature = pd.read_csv(csv_file,
                          header=0, encoding='utf-8',
                          usecols=[0,1,2,3,4,5,6,7,8,9,10,
                                11,12,13,14,15,16,17,18,19,20
                                ,21,22,23,24,25,26,27,28,29])
target = pd.read_csv(csv_file, header=0, encoding='utf-8',usecols=[30])

X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.1,random_state=1)

# from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# 1
# my_model = XGBRegressor()
# # Add silent=True to avoid printing out updates with each cycle
# my_model.fit(X_train, y_train, verbose=False)

#2
# my_model = XGBRegressor(n_estimators=1000)
# my_model.fit(X_train, y_train, early_stopping_rounds=5,
#              eval_set=[(X_test, y_test)], verbose=False)

#3
# my_model = XGBRegressor(objective ='reg:squarederror',max_depth = 3,n_estimators=100, learning_rate=0.1,eval_metric='rmse',
#                         colsample_bytree=0.8,subsample=0.8)
# my_model.fit(X_train, y_train, early_stopping_rounds=10,
#              eval_set=[(X_test, y_test)],verbose=True)

# my_model.fit(X_train, y_train, verbose=True)
# y_pred = my_model.predict(X_test)
forest = RandomForestRegressor(n_estimators=100,criterion='mse',random_state=1,n_jobs=1)

forest.fit(X_train,y_train)
y_pred = forest.predict(X_test)

print("MAPE: ",mean_absolute_percentage_error(y_test,y_pred))