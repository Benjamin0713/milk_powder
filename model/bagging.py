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
target = pd.read_csv(csv_file, header=0, encoding='utf-8',usecols=[41])

X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.1,random_state=1)

from sklearn.ensemble import BaggingRegressor

regr = BaggingRegressor(n_estimators=100,oob_score=True,random_state=1)

regr.fit(X_train,y_train)
y_pred = regr.predict(X_test)

print("MAPE: ",mean_absolute_percentage_error(y_test,y_pred))